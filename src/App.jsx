import Navbar from './components/Navbar';
import Hero from './components/Hero';
import Statistics from './components/Statistics';
import Features from './components/Features';
import WhyChoose from './components/WhyChoose';
import DemoPreview from './components/DemoPreview';
import Testimonials from './components/Testimonials';
import Footer from './components/Footer';

export default function App() {
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
