from pathlib import Path

from kb.eval.financial_reports_corpus import (
    build_english_seed_records,
    eastmoney_detail_url,
    eastmoney_pdf_url,
    eastmoney_record_matches,
    eastmoney_to_manifest_record,
    extract_dfcfw_cookie_header,
    fetch_eastmoney_page_with_retries,
    parse_jsonp,
)
from kb.eval.pdf_corpus import validate_record


def test_parse_jsonp_accepts_eastmoney_callback_shape():
    data = parse_jsonp('callback({"hits":1,"data":[{"infoCode":"AP1"}]})')

    assert data["hits"] == 1
    assert data["data"][0]["infoCode"] == "AP1"


def test_eastmoney_record_matches_semiconductor_keywords():
    assert eastmoney_record_matches({
        "title": "半导体设备行业深度报告",
        "industryName": "专用设备",
    })
    assert not eastmoney_record_matches({
        "title": "食品饮料行业周报",
        "industryName": "白酒",
    })


def test_eastmoney_urls_are_stable_from_info_code():
    assert eastmoney_pdf_url("AP202506091687511061").endswith(
        "/H3_AP202506091687511061_1.pdf"
    )
    assert "infocode=AP202506091687511061" in eastmoney_detail_url("AP202506091687511061")


def test_eastmoney_to_manifest_record_is_valid(tmp_path):
    record = eastmoney_to_manifest_record(
        {
            "infoCode": "AP202506091687511061",
            "title": "半导体行业5月份月报",
            "orgSName": "东海证券",
            "orgName": "东海证券股份有限公司",
            "researcher": "方霁",
            "publishDate": "2025-06-09 00:00:00.000",
            "industryName": "半导体",
            "industryCode": "1030",
            "attachSize": 5000,
            "attachPages": 20,
            "sRatingName": "增持",
        },
        root=tmp_path,
    )

    assert validate_record(record, line_no=1) == []
    assert record["source_family"] == "eastmoney"
    assert record["language"] == "zh"
    assert record["meta"]["institution"] == "东海证券"


def test_eastmoney_to_manifest_record_preserves_download_failure(tmp_path):
    record = eastmoney_to_manifest_record(
        {
            "infoCode": "AP202506091687511061",
            "title": "半导体行业5月份月报",
            "orgSName": "东海证券",
            "industryName": "半导体",
            "publishDate": "2025-06-09 00:00:00.000",
            "_download_ok": False,
            "_download_message": "http_404:Not Found",
        },
        root=tmp_path,
    )

    assert record["ingest_status"] == "failed"
    assert "download=http_404:Not Found" in record["notes"]


def test_extract_dfcfw_cookie_header_from_pdf_challenge():
    html = (
        '<script>var e={WTKkN:509015773,bOYDu:912484509,'
        'dtzqS:function(a,n){return a+n},wyeCN:2017756744,'
        'pCQRM:function(a){return a()}};'
        'switch(n[e++]){case"3":t=a[_0x649a("0x7")](t,1501954048);continue}'
        "</script>"
    )

    assert extract_dfcfw_cookie_header(html) == (
        "__tst_status=3439257026#; EO_Bot_Ssid=1501954048"
    )


def test_build_english_seed_records_are_valid(tmp_path):
    records = build_english_seed_records(root=tmp_path, download=False, limit=3)

    assert len(records) == 3
    assert {record["language"] for record in records} == {"en"}
    assert all(validate_record(record, line_no=1) == [] for record in records)
    assert all(Path(record["local_path"]).as_posix().startswith(
        "datasets/financial_semiconductor_reports_v1/raw/en/"
    ) for record in records)


def test_fetch_eastmoney_page_with_retries_retries_transient_error(monkeypatch):
    import kb.eval.financial_reports_corpus as corpus

    calls = {"count": 0}

    def flaky_fetch(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise OSError("temporary network hiccup")
        return {"hits": 1, "data": []}

    monkeypatch.setattr(corpus, "fetch_eastmoney_page", flaky_fetch)

    data = fetch_eastmoney_page_with_retries(
        page_no=1,
        page_size=50,
        begin_date="2026-01-01",
        end_date="2026-05-27",
        attempts=2,
        retry_delay_s=0,
    )

    assert data["hits"] == 1
    assert calls["count"] == 2
