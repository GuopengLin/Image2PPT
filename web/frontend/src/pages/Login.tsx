import { useState } from "react";
import { api, auth, type Me } from "../api/client";
import { Icon } from "../components/icons";
import logoUrl from "../assets/logo.png";

const REPO_URL = "https://github.com/GuopengLin/Image2PPT";

export default function Login({ onLogin }: { onLogin: (me: Me) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!password) return;
    setBusy(true);
    setErr(null);
    try {
      const res = await api.login(username, password);
      auth.setToken(res.access_token);
      onLogin({ id: 0, username: res.username, is_admin: res.is_admin });
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="login-wrap">
      <form className="login-card" onSubmit={submit}>
        <img className="login-logo" src={logoUrl} alt="DeckWeaver" />
        <h1>DeckWeaver</h1>
        <div className="sub">把图片或 PDF 转为可编辑 PowerPoint。</div>

        <label className="field">
          用户名
          <input
            className="input"
            autoFocus
            value={username}
            onChange={(e) => setUsername(e.target.value)}
          />
        </label>
        <label className="field">
          密码
          <input
            className="input"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
          />
        </label>

        {err && <div className="err">{err}</div>}

        <button className="btn primary login-btn" disabled={busy || !password}>
          {busy ? "登录中…" : "登录"}
        </button>

        <a className="login-repo" href={REPO_URL} target="_blank" rel="noreferrer">
          <Icon.Github />
          <div className="info">
            <div className="login-repo-name">GuopengLin/Image2PPT</div>
            <div className="login-repo-meta">查看源码 · 报告问题</div>
          </div>
          <span className="login-repo-arrow">↗</span>
        </a>

        <div className="foot">
          <span>deckweaver/web</span>
          <span>v0.1.0</span>
        </div>
      </form>
    </div>
  );
}
