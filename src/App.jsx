import { Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Statistics from './components/Statistics';
import Features from './components/Features';
import WhyChoose from './components/WhyChoose';
import DemoPreview from './components/DemoPreview';
import Testimonials from './components/Testimonials';
import Footer from './components/Footer';
import ChatPage from './components/ChatPage';

function LandingPage() {
  return (
    <div className="app">
      <Navbar />
      <Hero />
      <Statistics />
      <Features />
      <WhyChoose />
      <DemoPreview />
      <Testimonials />
      <Footer />
    </div>
  );
}

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/chat" element={<ChatPage />} />
    </Routes>
  );
}

