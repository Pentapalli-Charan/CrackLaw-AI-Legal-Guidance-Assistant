import { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Scale, ArrowRight } from 'lucide-react';
import './Navbar.css';

const navLinks = [
  { label: 'Home', href: '#home' },
  { label: 'Features', href: '#features' },
  { label: 'AI Assistant', href: '/chat', isRoute: true },
  { label: 'Legal Research', href: '#why-choose' },
  { label: 'About', href: '#testimonials' },
];

export default function Navbar() {
  const navigate = useNavigate();
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50);
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    document.body.style.overflow = mobileOpen ? 'hidden' : '';
    return () => { document.body.style.overflow = ''; };
  }, [mobileOpen]);

  return (
    <>
      <motion.nav
        className={`navbar ${scrolled ? 'scrolled' : ''}`}
        initial={{ y: -80 }}
        animate={{ y: 0 }}
        transition={{ duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
      >
        <div className="container navbar-inner">
          <a href="#home" className="navbar-logo">
            <div className="navbar-logo-icon">
              <Scale size={20} color="#ffffff" strokeWidth={2.5} />
            </div>
            <span>CrackLaw</span>
          </a>

          <div className="navbar-links">
            {navLinks.map((link) =>
              link.isRoute ? (
                <a key={link.label} href={link.href} className="navbar-link" onClick={(e) => { e.preventDefault(); navigate(link.href); }}>
                  {link.label}
                </a>
              ) : (
                <a key={link.label} href={link.href} className="navbar-link">
                  {link.label}
                </a>
              )
            )}
          </div>

          <button className="btn-primary navbar-cta desktop" onClick={() => navigate('/chat')}>
            Start AI Consultation <ArrowRight size={16} />
          </button>

          <button
            className={`navbar-mobile-toggle ${mobileOpen ? 'open' : ''}`}
            onClick={() => setMobileOpen(!mobileOpen)}
            aria-label="Toggle menu"
          >
            <span />
            <span />
            <span />
          </button>
        </div>
      </motion.nav>

      <div className={`navbar-mobile-menu ${mobileOpen ? 'open' : ''}`}>
        {navLinks.map((link) => (
          <a
            key={link.label}
            href={link.href}
            className="navbar-link"
            onClick={(e) => { if (link.isRoute) { e.preventDefault(); navigate(link.href); } setMobileOpen(false); }}
          >
            {link.label}
          </a>
        ))}
        <button className="btn-primary navbar-cta" onClick={() => { navigate('/chat'); setMobileOpen(false); }}>
          Start AI Consultation <ArrowRight size={16} />
        </button>
      </div>
    </>
  );
}
