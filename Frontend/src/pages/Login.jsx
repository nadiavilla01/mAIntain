import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { FiMail, FiLock } from "react-icons/fi";
import logo from "../assets/logo.png"; 
import "./Login.css";

export default function Login() {
  const nav = useNavigate();
  const [showPwd, setShowPwd] = useState(false);

  const onSubmit = (e) => {
    e.preventDefault();
    nav("/");
  };

  return (
    <div className="login-wrap is-vertical">
      <div className="grid-tex" />
      <div className="diag-lines" />
      <div className="neon-haze tl" />
      <div className="neon-haze br" />
      <div className="sweep-beam" />

      <div className="login-shell vertical">
        <section className="hero">
          <div className="hero-glow" />
          <img src={logo} className="brand huge" alt="AI.ndustry" />
        </section>

        <section className="auth-card">
          <div className="topline" />
          <h1>Welcome back</h1>
          <p className="muted">Sign in to your AI.ndustry console</p>

          <form onSubmit={onSubmit}>
            <label className="input">
              <FiMail />
              <input type="email" placeholder="you@company.com" required />
            </label>

            <label className="input">
              <FiLock />
              <input
                type={showPwd ? "text" : "password"}
                placeholder="Password"
                required
              />
              <button
                type="button"
                className="peek"
                onClick={() => setShowPwd((s) => !s)}
                aria-label="Show password"
                title="Show password"
              >
                {showPwd ? "🙈" : "👁️"}
              </button>
            </label>

            <div className="row between">
              <label className="check">
                <input type="checkbox" /> <span>Remember me</span>
              </label>
              <button type="button" className="text-btn">
                Forgot password?
              </button>
            </div>

            <button className="cta" type="submit">
              Sign in
            </button>
          </form>

          <p className="tiny-tip">Tip: this is a demo login — no backend required.</p>
        </section>
      </div>
    </div>
  );
}
