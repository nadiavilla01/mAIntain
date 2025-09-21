import React, { useEffect, useState } from 'react';
import './Loader.css';
import logo from '../assets/logo.png';

const Loader = () => {
  const [fadeOut, setFadeOut] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setFadeOut(true);
    }, 3000); 
    return () => clearTimeout(timer);
  }, []);

  return (
    <div className={`loader-screen ${fadeOut ? 'fade-out' : ''}`}>
      <div className="particles-bg" />
      <img src={logo} alt="AI.ndustry logo" className="loader-logo" />
      <p className="loader-subtext">Launching predictive intelligence…</p>
      <div className="loader-glow-bar" />
    </div>
  );
};

export default Loader;
