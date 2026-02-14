```bash
docker compose up -d

docker exec -it rag_pgvector psql -U postgres -d rag_production
```


---

## Useful psql commands (inside container)

```sql
-- List databases
\l

-- Use rag_production
\c rag_production

-- List extensions
\dx

-- List tables
\dt

-- List LangChain tables
\dt langchain*

-- Row count
SELECT COUNT(*) FROM langchain_pg_embedding;

-- Sample metadata (JSONB)
SELECT cmetadata->>'source_file', cmetadata->>'department' FROM langchain_pg_embedding LIMIT 3;

-- Quit
\q
```

---