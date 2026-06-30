import { useRef } from 'react';
import { motion } from 'framer-motion';
import { ChevronLeft, ChevronRight, Star } from 'lucide-react';
import './Testimonials.css';

const testimonials = [
  {
    name: 'Adv. Priya Sharma',
    role: 'Corporate Lawyer, Mumbai',
    avatar: 1,
    initials: 'PS',
    stars: 5,
    text: 'CrackLaw has completely transformed my legal research workflow. What used to take hours now takes minutes. The <em>AI accuracy</em> is genuinely impressive — it cites relevant case laws I would have missed.',
  },
  {
    name: 'Rajesh Menon',
    role: 'Startup Founder, Bangalore',
    avatar: 2,
    initials: 'RM',
    stars: 5,
    text: 'As a non-lawyer, navigating contracts was overwhelming. CrackLaw\'s <em>contract review feature</em> helped me understand every clause in my investment agreement. It\'s like having a legal advisor on speed dial.',
  },
  {
    name: 'Adv. Meera Kapoor',
    role: 'Litigation Specialist, Delhi',
    avatar: 3,
    initials: 'MK',
    stars: 5,
    text: 'The judgment summaries are incredibly accurate. I use CrackLaw daily to prepare for hearings — the <em>precedent analysis</em> alone saves me hours of manual research.',
  },
  {
    name: 'Arjun Patel',
    role: 'Legal Compliance, Ahmedabad',
    avatar: 4,
    initials: 'AP',
    stars: 4,
    text: 'Our compliance team relies on CrackLaw for regulatory research. The <em>risk assessment tool</em> helps us identify potential issues before they become problems. Essential for any legal department.',
  },
  {
    name: 'Dr. Sanjana Roy',
    role: 'Law Professor, Kolkata',
    avatar: 5,
    initials: 'SR',
    stars: 5,
    text: 'I recommend CrackLaw to all my students. The way it <em>breaks down complex statutes</em> into understandable language is remarkable. It\'s the future of legal education and practice.',
  },
];

export default function Testimonials() {
  const trackRef = useRef(null);

  const scroll = (dir) => {
    if (!trackRef.current) return;
    const amount = 400;
    trackRef.current.scrollBy({ left: dir === 'left' ? -amount : amount, behavior: 'smooth' });
  };

  return (
    <section className="testimonials section" id="testimonials">
      <div className="container">
        <motion.div
          className="testimonials-header"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <span className="section-label">Testimonials</span>
          <h2 className="section-title">Trusted by Legal Professionals</h2>
          <p className="section-subtitle">
            Hear from lawyers, entrepreneurs, and legal scholars who rely on CrackLaw
            for accurate, instant legal intelligence.
          </p>
        </motion.div>

        <motion.div
          className="testimonials-track-wrapper"
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <div className="testimonials-track" ref={trackRef}>
            {testimonials.map((t) => (
              <div key={t.name} className="testimonial-card glass-card">
                <div className="testimonial-stars">
                  {Array.from({ length: t.stars }).map((_, i) => (
                    <Star key={i} size={14} fill="#c9a84c" className="testimonial-star" />
                  ))}
                </div>
                <p className="testimonial-text" dangerouslySetInnerHTML={{ __html: t.text }} />
                <div className="testimonial-author">
                  <div className={`testimonial-avatar testimonial-avatar-${t.avatar}`}>
                    {t.initials}
                  </div>
                  <div className="testimonial-author-info">
                    <h4>{t.name}</h4>
                    <span>{t.role}</span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        <div className="testimonials-nav">
          <button onClick={() => scroll('left')} aria-label="Previous">
            <ChevronLeft size={20} />
          </button>
          <button onClick={() => scroll('right')} aria-label="Next">
            <ChevronRight size={20} />
          </button>
        </div>
      </div>
    </section>
  );
}
