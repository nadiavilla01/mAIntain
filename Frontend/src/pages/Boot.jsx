import React, { useEffect } from "react";
import { useNavigate } from "react-router-dom";
import Loader from "../components/Loader";

export default function Boot() {
  const nav = useNavigate();
  useEffect(() => {
    const t = setTimeout(() => nav("/"), 1800); 
    return () => clearTimeout(t);
  }, [nav]);
  return <Loader />; 
}
