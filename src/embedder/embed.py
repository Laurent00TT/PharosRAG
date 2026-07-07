"""索引期:消费 chunker 的 ChunkResult + 原始 elements,向量化并写入 Qdrant + sidecar。

分流(对应 chunker 的 image_only flag):
  - image_only 纯图  -> 图像向量(dense),**跳过 sparse**(纯图无可检索文字,靠图文同空间召回);
  - text/table/有文字的 image/chart -> 文字 dense + BM25 sparse。
ACL:`acl_split` 把 chunk.acl 拆成 4 个可过滤 payload 字段(fail-closed)。
sidecar:per doc_id 存 elements+sections+banners+acl_index —— retrieve 期 assemble_big 的 small-to-big 必需
(Qdrant 只存 chunk 向量,原始 element/section 不进库)。"""
from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import asdict

from qdrant_client import models

from .acl import acl_split
from .config import SIDECAR_VERSION, EmbedConfig
from .dense import Dense
from .remote import make_dense
from .sparse import doc_sparse
from .store import Store


def _payload(ch, acl_fields: dict) -> dict:
    """进 Qdrant 的 payload:展示/引用/过滤 + retrieve 重建 hit chunk 所需字段 + ACL。"""
    return {
        "chunk_id": ch.chunk_id, "doc_id": ch.doc_id, "kind": ch.kind, "text": ch.text,
        "content_raw": ch.content_raw, "breadcrumb": ch.breadcrumb, "section_path": ch.section_path,
        "section_id": ch.section_id, "section_anchor": ch.section_anchor,
        "page_start": ch.page_start, "page_end": ch.page_end, "source_indices": ch.source_indices,
        "flags": ch.flags, "lang": ch.lang, "doc_type": ch.doc_type,
        "image_path": ch.image_path, "doc_meta": ch.doc_meta,
        "acl": ch.acl,                       # 原始 acl 也存:出口二次校验(铁律5)/ 调试
        **acl_fields,                        # acl_unset/acl_tenant/acl_allow/acl_visibility(可过滤)
    }


class Embedder:
    def __init__(self, cfg: EmbedConfig | None = None, store: Store | None = None, dense: Dense | None = None):
        # store/dense 可共享:嵌入式 Qdrant 同一 path 只允许一个 client(单进程内 embed+retrieve 必须共用),
        # 且复用 dense 避免重复 load 8B 模型(省 ~2min + 显存)。生产 server 模式无此 client 约束。
        self.cfg = cfg or EmbedConfig()
        self.dense = dense or make_dense(self.cfg)   # 工厂:cfg.inference_url 空=local(默认),非空=远程推理服务
        self.store = store or Store(self.cfg)
        self.store.ensure_collection()
        os.makedirs(self.cfg.sidecar_dir, exist_ok=True)

    @staticmethod
    def _pid(chunk_id: str) -> str:
        # Qdrant point id 需 uint64/UUID;chunk_id 是字符串 -> 确定性 UUID5(重跑同 id => upsert 覆盖,幂等)
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    def index_document(self, doc_id: str, elements: list, chunk_result, image_root: str) -> dict:
        """elements: 原始 list[Element](idx 对齐);chunk_result: ChunkResult;
        image_root: MinerU 输出根目录(拼 image_path 绝对路径,= parsed/<doc>/)。

        **契约:doc_id 必须全局唯一且稳定**。point_id = uuid5(chunk_id),chunk_id = f"{doc_id}#{n}";
        同 doc_id 重索引是幂等覆盖(feature),但**不同内容复用同 doc_id 会静默覆盖前者的向量+payload+ACL+sidecar**
        (seal#3)。多源 ingest 须各自保证 doc_id 命名空间不撞(如加源前缀)。

        失败面(评审修):编码 + sidecar tmp 都是**纯准备**,失败无副作用(旧索引原样可用);
        delete->upsert->replace 收尾段毫秒级,该窗口内失败=该 doc 脱库(旧向量已删、新数据未落),
        响亮告警后 raise,必须重跑本方法恢复 —— 不再是旧实现"先删后逐 chunk 编码"的分钟级脱库窗口。"""
        points, n_img, n_txt, n_skip = [], 0, 0, 0
        for ch in chunk_result.chunks:
            af = acl_split(ch.acl)
            if "image_only" in (ch.flags or []):
                if not ch.image_path:                          # 纯图但无路径 -> 无法向量化
                    n_skip += 1; continue
                p = os.path.join(image_root, ch.image_path)
                if not os.path.exists(p):                       # 路径失效(净化后/移动)-> 跳过,不静默假装
                    n_skip += 1; continue
                dvec, svec = self.dense.encode_image([p])[0], None
                n_img += 1
            else:
                dvec = self.dense.encode_text([ch.text])[0]
                svec = doc_sparse(ch.text, self.cfg.stopwords)
                n_txt += 1
            vec: dict = {"dense": dvec.tolist()}
            if svec is not None:
                vec["sparse"] = svec
            points.append(models.PointStruct(id=self._pid(ch.chunk_id), vector=vec, payload=_payload(ch, af)))
        # sidecar tmp 也在 delete 前备好(json.dump+fsync 是磁盘满等失败的主要发生点),delete 后只剩原子改名
        tmp, path = self._prepare_sidecar(doc_id, elements, chunk_result)
        deleted = False
        try:
            # 删旧仍在 upsert 前:重索引若变短,旧高编号 point 不会被覆盖 -> 删旧防孤儿(review#4 动机不变)
            self.store.delete_by_doc(doc_id)
            deleted = True
            if points:
                self.store.upsert(points)
            os.replace(tmp, path)          # 同盘原子改名:新向量与新 sidecar 同代落地(seal#9)
        except Exception as e:
            try:
                os.remove(tmp)             # 清理准备产物;失败不掩盖主异常
            except OSError:
                pass
            if deleted:                    # 旧向量已删、新数据未写全 —— 静默 raise 会让"跳过"掩盖脱库事实
                print(f"[embedder] FATAL: doc {doc_id} 已脱库(旧向量已删,新向量/sidecar 未落库):"
                      f"{type(e).__name__}: {e} —— 必须重新 index_document 恢复。", file=sys.stderr, flush=True)
            raise
        return {"images": n_img, "texts": n_txt, "skipped": n_skip, "indexed": len(points)}

    def _prepare_sidecar(self, doc_id: str, elements: list, chunk_result) -> tuple[str, str]:
        """把 sidecar 全量写到 .tmp + fsync,返回 (tmp, path)。**不落正式文件**:调用方在库写成功后
        os.replace 收尾。准备/收尾拆开是 index_document 失败面收窄的一半 —— 写盘失败发生在 delete 之前
        (无副作用),delete 之后只剩同盘原子改名(seal#9:防写一半截断 JSON 让 small-to-big 永久崩)。"""
        data = {
            "version": SIDECAR_VERSION,        # 读侧校验:不符即拒,提示重建(schema 漂移防静默错)
            "elements": [asdict(e) for e in elements],
            "sections": [asdict(s) for s in chunk_result.sections],
            "banners": list(chunk_result.banners),
            # acl_index: {element_idx: acl} —— assemble_big 用它保证 small-to-big 不跨 ACL 取材
            "acl_index": {str(k): v for k, v in chunk_result.acl_index().items()},
        }
        path = os.path.join(self.cfg.sidecar_dir, f"{doc_id}.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        return tmp, path

    def _write_sidecar(self, doc_id: str, elements: list, chunk_result) -> None:
        tmp, path = self._prepare_sidecar(doc_id, elements, chunk_result)
        os.replace(tmp, path)

    def delete_document(self, doc_id: str) -> None:
        """删一个文档:Qdrant point + sidecar 同删,避免只删一边留孤儿(孤儿向量可召回 / 孤儿 sidecar 占盘)。"""
        self.store.delete_by_doc(doc_id)
        path = os.path.join(self.cfg.sidecar_dir, f"{doc_id}.json")
        if os.path.exists(path):
            os.remove(path)
