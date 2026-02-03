# 面试数据目录

> **最后更新**：2026-02-02
> **用途**：存储面试相关的所有数据（音频、转录文本、面经文档）

---

## 目录结构

```
interview_data/
├── audio/              # 原始音频文件
│   ├── 20260215_字节_AI前端.mp3
│   └── 20260220_小红书_Agent研发.m4a
├── transcripts/        # 转录文本
│   ├── 20260215_字节_AI前端_transcript.md
│   └── 20260220_小红书_Agent研发_transcript.md
├── summaries/          # 面经文档
│   ├── 20260215_字节_AI前端.md
│   └── 20260220_小红书_Agent研发.md
├── metadata.json       # 元数据索引
└── README.md           # 本文件
```

---

## 文件命名规范

### 音频文件（audio/）
```
格式：YYYYMMDD_公司_岗位.扩展名
示例：20260215_字节_AI前端.mp3
```

### 转录文本（transcripts/）
```
格式：YYYYMMDD_公司_岗位_transcript.md
示例：20260215_字节_AI前端_transcript.md
```

### 面经文档（summaries/）
```
格式：YYYYMMDD_公司_岗位.md
示例：20260215_字节_AI前端.md
```

---

## metadata.json 结构

```json
{
  "last_updated": "2026-02-02",
  "total_interviews": 2,
  "interviews": [
    {
      "date": "2026-02-15",
      "company": "字节跳动",
      "position": "AI前端工程师",
      "round": "1st",
      "status": "Waiting",
      "audio": "audio/20260215_字节_AI前端.mp3",
      "transcript": "transcripts/20260215_字节_AI前端_transcript.md",
      "summary": "summaries/20260215_字节_AI前端.md"
    }
  ]
}
```

---

## 使用方式

### 记录面试

使用interview-recorder skill：

```
用户: /记录面试

AI会引导你：
1. 输入基本信息（公司、岗位、轮次等）
2. 提供音频文件（如果有）
3. 自动生成转录文本和面经文档
4. 更新metadata.json
5. 更新08/09文档
```

### 查看面试记录

```
用户: /查看面试

AI会：
1. 读取metadata.json
2. 显示所有面试记录列表
3. 可以查看具体某个面试的详细信息
```

---

## 与其他系统的联动

- **08_Action_Plan_2026_H1.md**：面试记录会添加到"每周进度更新"
- **09_Progress_Tracker.md**：面试记录会添加到"三、每周进度更新"
- **interview-recorder skill**：自动维护本目录的所有文件

---

## 注意事项

1. **文件命名**：严格遵循命名规范，便于查找和管理
2. **音频存储**：音频文件可能较大，定期清理不需要的文件
3. **隐私保护**：音频文件和转录文本可能包含敏感信息，谨慎处理
4. **Git提交**：音频文件提交到Git前请确认是否需要（.gitignore已配置）

---

## 更新日志

| 日期 | 更新内容 |
|------|---------|
| 2026-02-02 | 初始版本，建立目录结构 |
