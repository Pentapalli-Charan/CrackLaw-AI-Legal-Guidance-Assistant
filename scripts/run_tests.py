import sys
import os
import unittest

# Add project root to sys.path to enable src imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

def run_all_tests():
    print("==================================================")
    print("           CRACKLAW TEST RUNNER                   ")
    print("==================================================")
    
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("==================================================")
    print("           TEST RUN SUMMARY                       ")
    print("==================================================")
    print(f"Tests Run:      {result.testsRun}")
    print(f"Failures:       {len(result.failures)}")
    print(f"Errors:         {len(result.errors)}")
    print("==================================================")
    
    if not result.wasSuccessful():
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    run_all_tests()
