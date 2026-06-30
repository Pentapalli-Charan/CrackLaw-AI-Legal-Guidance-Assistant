import { motion } from 'framer-motion';
import './Statistics.css';

const stats = [
  { number: '95%', label: 'Legal Query Accuracy' },
  { number: '1.5M+', label: 'Legal Documents Analyzed' },
  { number: '50K+', label: 'Court Judgments' },
  { number: '150+', label: 'Legal Acts Covered' },
];

const container = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.12 } },
};

const item = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.5, ease: [0.4, 0, 0.2, 1] } },
};

export default function Statistics() {
  return (
    <section className="statistics section">
      <div className="container">
        <motion.div
          className="statistics-grid"
          variants={container}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.3 }}
        >
          {stats.map((stat) => (
            <motion.div
              key={stat.label}
              className="stat-card glass-card"
              variants={item}
              whileHover={{ scale: 1.04, y: -4 }}
              transition={{ type: 'spring', stiffness: 300 }}
            >
              <div className="stat-number">{stat.number}</div>
              <div className="stat-label">{stat.label}</div>
            </motion.div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
