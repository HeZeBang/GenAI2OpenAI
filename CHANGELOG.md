# Changelog

## v2.0.0

### Breaking Changes

- `--token` 参数现在为必需项，移除了硬编码的默认 token
- `Config.token` 字段替换为 `Config.token_manager`，使用 `TokenManager` 对象管理 token 生命周期
- 新增 `pycryptodome` 依赖

### New Features

- **学号密码登录**: `--token` 支持 `学号@密码` 格式，自动通过 CAS 统一身份认证获取 JWT
- **Token 自动刷新**: 学号密码模式下，JWT 过期时自动重新登录，对客户端完全透明
- **401 自动重试**: 上游返回 401 时，自动刷新 token 并重试当前请求
- **JWT 离线校验**: 解码 JWT payload 中的 `exp` 字段，预留 60 秒安全余量提前刷新
- **启动时快速失败**: 学号密码模式启动时立即尝试登录，密码错误直接报错退出

### Internal

- 新增 `auth/cas_login.py`: CAS 登录流程（AES-128-CBC 密码加密、IDS 表单解析、重定向跟随）
- 新增 `auth/token_manager.py`: `TokenManager` 类（JWT/学号密码模式识别、线程安全刷新）
- `app.py`: `before_request` 改用 `token_manager.get_token()`，新增 `LoginError` 错误处理
- `provider/genai.py`: 401 响应时调用 `force_refresh()` 并重试

## v1.0.0

### Features

- OpenAI 兼容的 Chat Completion 代理（流式/非流式）
- 动态模型列表，自动从 GenAI 平台拉取
- Tool Calling 支持（通过 prompt 注入，兼容非原生模型）
- 流式 Tool Calling 解析
- Token 过期流式错误推送
- API Key 客户端认证
- 健康检查端点
