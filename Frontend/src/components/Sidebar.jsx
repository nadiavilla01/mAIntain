import React, { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import {
  AiOutlineDashboard,
  AiOutlineQuestionCircle,
} from "react-icons/ai";
import { GiFactory } from "react-icons/gi";
import { FiBarChart2, FiMoon, FiSun } from "react-icons/fi";
import { MdOutlineHistory, MdSettings } from "react-icons/md";
import { RiErrorWarningLine } from "react-icons/ri";
import {
  LuThermometer,
  LuVibrate,
  LuZap,
  LuGauge,
  LuChevronLeft,
} from "react-icons/lu";
import "./Sidebar.css";
import logo from "../assets/logo.png";

export default function Sidebar() {
  const { pathname } = useLocation();
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState(false);
  const [dark, setDark] = useState(true);

  const links = [
    { path: "/",              label: "Dashboard",      icon: <AiOutlineDashboard size={20} /> },
    { path: "/machines",      label: "Machines",       icon: <GiFactory size={20} /> },
    { path: "/insights",      label: "AI Insights",    icon: <FiBarChart2 size={20} /> },
    { path: "/fault-detection", label: "Fault Detection", icon: <RiErrorWarningLine size={20} /> },
    { path: "/history",       label: "History",        icon: <MdOutlineHistory size={20} /> },
    { path: "/settings",      label: "Settings",       icon: <MdSettings size={20} /> },
  ];


  const toggleTheme = () => setDark((d) => !d);

  return (
    <aside
      className={`sidebar ${collapsed ? "is-collapsed" : ""}`}
      role="navigation"
      aria-label="Primary"
    >
      <div className="logo">
        <img src={logo} alt="AI.ndustry" className="sidebar-logo" />
      </div>

      

      <nav className="menu">
        {links.map(({ path, label, icon }) => {
          const isActive = pathname === path;
          return (
            <Link
              key={path}
              to={path}
              className={`menu-link ${isActive ? "active" : ""}`}
              aria-current={isActive ? "page" : undefined}
            >
              {icon}
              {!collapsed && <span>{label}</span>}
            </Link>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <button
          className="fbtn"
          onClick={toggleTheme}
          title={dark ? "Light mode" : "Dark mode"}
          aria-label="Toggle theme"
        >
          {dark ? <FiSun size={18} /> : <FiMoon size={18} />}
        </button>
        <button
          className="fbtn"
          onClick={() => navigate("/insights")}
          title="Help / Assistant"
          aria-label="Help"
        >
          <AiOutlineQuestionCircle size={18} />
        </button>
        <button
          className="fbtn f-collapse"
          onClick={() => setCollapsed((c) => !c)}
          title={collapsed ? "Expand" : "Collapse"}
          aria-label="Collapse sidebar"
        >
          <LuChevronLeft size={18} />
        </button>
      </div>
    </aside>
  );
}
