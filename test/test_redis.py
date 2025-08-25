import redis

r = redis.Redis(host="redis", port=6379, db=0)

def test1():
    job_id = "job123"
    r.hset(f"OCRJob:{job_id}", mapping={
        "url": "http://example.com/file.pdf",
        "knowledgebaseId": "kb001",
        "workspaceId": "ws001",
        "markdownUrl": "http://example.com/file.md",
        "jsonUrl": "http://example.com/file.json",
        "status": "pending"
    })

    record = r.hgetall(f"OCRJob:{job_id}")
    record_str = {k.decode(): v.decode() for k, v in record.items()}
    print(record_str)

def test2():
    for key in r.scan_iter("OCRJobId:*"):
        record = r.hgetall(key)
        record_trans = {k.decode(): v.decode() for k, v in record.items()}
        print(key)
        print(record_trans)

if __name__ == "__main__":
    test2()