import { motion } from 'framer-motion';
import {
  Zap, Clock, ShieldCheck, Brain, Globe, Award
} from 'lucide-react';
import './WhyChoose.css';

const benefits = [
  {
    icon: Brain,
    title: 'AI-Powered Accuracy',
    desc: 'Our AI is trained on millions of legal documents, delivering precise and contextually relevant results.',
  },
  {
    icon: Clock,
    title: '24/7 Availability',
    desc: 'Get legal insights anytime, anywhere — no appointments, no waiting rooms, no office hours.',
  },
  {
    icon: ShieldCheck,
    title: 'Bank-Grade Security',
    desc: 'End-to-end encryption and SOC 2 compliance ensure your confidential legal data stays protected.',
  },
  {
    icon: Zap,
    title: 'Instant Case Analysis',
    desc: 'Analyze complex cases in seconds with AI that understands legal context, precedents, and nuances.',
  },
  {
    icon: Globe,
    title: 'Multi-Jurisdiction Support',
    desc: 'Comprehensive coverage across Indian legal codes, international law, and specialized regulatory frameworks.',
  },
];

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.4, 0, 0.2, 1] } },
};

export default function WhyChoose() {
  return (
    <section className="why-choose section" id="why-choose">
      <div className="container why-choose-inner">
        {/* Visual Side */}
        <motion.div
          className="why-choose-visual"
          initial={{ opacity: 0, x: -40 }}
          whileInView={{ opacity: 1, x: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6, ease: [0.4, 0, 0.2, 1] }}
        >
          <div className="why-visual-bg" />
          <div className="why-visual-lines">
            <div className="why-visual-line" />
            <div className="why-visual-line" />
            <div className="why-visual-line" />
          </div>
          <div className="why-visual-circle" />
          <div className="why-visual-circle" />
          <div className="why-visual-circle" />
          <div className="why-visual-scale">⚖</div>
          <div className="why-visual-badge">
            <div className="why-visual-badge-icon">
              <Award size={18} color="#ffffff" />
            </div>
            <div className="why-visual-badge-text">
              <strong>Trusted Platform</strong>
              <span>50,000+ Legal Professionals</span>
            </div>
          </div>
        </motion.div>

        {/* Content Side */}
        <motion.div
          className="why-choose-content"
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.2 }}
          transition={{ staggerChildren: 0.1 }}
        >
          <motion.div variants={fadeUp}>
            <span className="section-label">Why CrackLaw</span>
            <h2 className="section-title">Built for Legal Excellence</h2>
            <p className="section-subtitle">
              CrackLaw combines the precision of legal expertise with the speed of artificial
              intelligence to transform how you approach legal challenges.
            </p>
          </motion.div>

          <div className="why-choose-list">
            {benefits.map((b, i) => (
              <motion.div
                key={b.title}
                className="why-choose-item"
                variants={fadeUp}
                custom={i}
              >
                <div className="why-choose-item-icon">
                  <b.icon size={20} />
                </div>
                <div className="why-choose-item-text">
                  <h4>{b.title}</h4>
                  <p>{b.desc}</p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  );
}
