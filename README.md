# codey

Production-grade multi-agent AI code review system.

[LangGraph](https://github.com/langchain-ai/langgraph) supervisor orchestrating specialized agents — security, code quality, testing, and indexing — over your local git diff, with tree-sitter AST caching and jedi-based reverse-dependency lookup.

```
codey set      # configure provider & API key (OpenAI / Anthropic / DeepSeek / Google / custom)
codey model    # view or switch the active model
codey config   # show current configuration
codey review   # review the latest commit in the local repo
```

## License

MIT