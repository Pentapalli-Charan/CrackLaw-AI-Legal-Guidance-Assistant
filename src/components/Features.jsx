import { motion } from 'framer-motion';
import {
  MessageSquare, Search, FileText,
  FileCheck, BookOpen, ShieldAlert,
} from 'lucide-react';
import './Features.css';

const features = [
  {
    icon: MessageSquare,
    title: 'AI Legal Chat',
    desc: 'Get instant answers to complex legal queries with our advanced AI trained on millions of legal documents and precedents.',
  },
  {
    icon: Search,
    title: 'Legal Research',
    desc: 'Search through comprehensive databases of case laws, statutes, and legal opinions with AI-powered semantic understanding.',
  },
  {
    icon: FileText,
    title: 'Document Analysis',
    desc: 'Upload legal documents and receive detailed AI analysis highlighting key clauses, risks, and compliance requirements.',
  },
  {
    icon: FileCheck,
    title: 'Contract Review',
    desc: 'Automated contract review that identifies potential issues, unfavorable terms, and suggests improvements in seconds.',
  },
  {
    icon: BookOpen,
    title: 'Judgment Summary',
    desc: 'Get concise, accurate summaries of court judgments with key takeaways, precedent analysis, and ratio decidendi.',
  },
  {
    icon: ShieldAlert,
    title: 'Risk Assessment',
    desc: 'Evaluate legal risks for business decisions with AI-powered analysis of regulatory frameworks and compliance standards.',
  },
];

const container = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.1 } },
};

const item = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.4, 0, 0.2, 1] } },
};

export default function Features() {
  return (
    <section className="features section" id="features">
      <div className="container">
        <motion.div
          className="features-header"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          <span className="section-label">Features</span>
          <h2 className="section-title">Powerful AI Legal Tools</h2>
          <p className="section-subtitle">
            Everything you need to navigate the legal landscape — powered by cutting-edge
            artificial intelligence and comprehensive legal databases.
          </p>
        </motion.div>

        <motion.div
          className="features-grid"
          variants={container}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.2 }}
        >
          {features.map((f) => (
            <motion.div
              key={f.title}
              className="feature-card glass-card"
              variants={item}
              whileHover={{ y: -6 }}
              transition={{ type: 'spring', stiffness: 300 }}
            >
              <div className="feature-icon">
                <f.icon size={24} />
              </div>
              <h3>{f.title}</h3>
              <p>{f.desc}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
