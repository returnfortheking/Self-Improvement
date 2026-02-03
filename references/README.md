# References - External Learning Resources

> **Purpose**: Store high-quality external resources (GitHub repos, tech blogs) for learning material generation
> **Last Updated**: 2026-02-03
> **Maintained by**: doc-sync Skill

---

## 📁 Directory Structure

```
references/
├── github/               # Cloned GitHub repositories
│   ├── python-interview/ # Python interview questions
│   ├── llm-interview/    # LLM/RAG/Agent interview resources
│   └── .metadata.json    # Repository metadata
│
├── tech-blogs/           # Crawled tech blog articles
│   ├── aliyun/           # 阿里云技术博客
│   ├── tencent/          # 腾讯技术
│   ├── meituan/          # 美团技术团队
│   ├── bytedance/        # 字节技术团队
│   └── .metadata.json    # Blog metadata
│
└── metadata/             # Index and configuration
    ├── content_index.json        # ⭐ Content index (incremental scanning)
    ├── repos_to_sync.json        # Repositories to sync
    ├── blogs_to_sync.json        # Blogs to sync
    ├── quality_rules.json        # Quality filtering rules
    └── last_sync.json            # Last sync timestamp
```

---

## 🔄 Update Mechanism

### Manual Trigger
```bash
# Update all resources
/更新资源

# Sync specific repository
/更新资源 baliyanvinay/Python-Interview-Preparation

# Rebuild index from scratch
/重建索引
```

### Automatic Trigger
- After `doc-sync` detects significant new resources (>50 new topics)
- Weekly check (can be configured)

---

## 📊 Space Usage

**Limit**: 10GB for GitHub repositories

**Current Usage**: (Check with `du -sh references/`)

---

## 🔍 Content Index

The `content_index.json` file is the heart of the incremental scanning system:

- **First scan**: Full scan of all files (~20 minutes)
- **Subsequent scans**: Only process changed files (<1 minute, 98.6% files skipped)
- **File hashing**: SHA-256 for change detection
- **Topics extraction**: Automatic topic and question extraction
- **Quality scoring**: Based on source quality and content metrics

---

## 📋 Resource Quality Criteria

### GitHub Repositories
- Min stars: 100
- Max inactive days: 180
- Prefer Chinese documentation
- Active maintenance

### Tech Blog Articles
- Min views: 500-1500 (varies by source)
- Min likes: 20-50
- Must have code or diagrams
- Exclude ads and promotions

---

## 🛠️ Maintenance

### Adding New Repositories
Edit `metadata/repos_to_sync.json`:
```json
{
  "github_repos": [
    {
      "name": "your-repo-name",
      "url": "https://github.com/user/repo.git",
      "category": "category-name",
      "enabled": true
    }
  ]
}
```

### Adding New Blogs
Edit `metadata/blogs_to_sync.json`:
```json
{
  "tech_blogs": [
    {
      "company": "Company Name",
      "base_url": "https://example.com/",
      "quality_filter": {
        "min_views": 1000,
        "exclude_keywords": ["ad", "promotion"]
      },
      "enabled": true
    }
  ]
}
```

---

## ⚠️ Important Notes

1. **Do NOT manually edit** `content_index.json` - it's auto-generated
2. **Do NOT manually edit** cloned repositories - they will be overwritten on next sync
3. **All content is read-only** - used for learning material generation only
4. **Quality filtering is automatic** - only high-quality content is indexed

---

## 📈 Statistics

(After first sync)

**GitHub Repositories**: X repos, Y files
**Tech Blog Articles**: X articles from Y companies
**Total Indexed Topics**: Z topics
**Content Index Size**: ~625 KB

---

**Last Sync**: 2026-02-03 (Initial)
**Next Sync**: Manual trigger or automatic (weekly)
