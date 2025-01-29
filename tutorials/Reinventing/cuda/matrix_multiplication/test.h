#include <cstdlib>
#include <cmath>
#include <iostream>
#include <string>

void testResult(const float* h_C, const float* C_ref, int ROWS_A, int COLS_B, bool ELEMENT_WISE, const std::string& kernelName) {
  // Verify results
  std::cout <<"\n \t\t\t\t" << "_________ " << kernelName << " Result ___________ \n";
  bool testPassed = true;
  for (int row = 0; row < ROWS_A; row++) {
    for (int col = 0; col < COLS_B; col++) {
      int index = row * COLS_B + col;
      if (ELEMENT_WISE) {
        std::cout << h_C[index] << " (";
        if (std::abs(h_C[index] - C_ref[index]) < 1e-5) {
          std::cout << "PASS";
        } else {
          std::cout << "FAIL, expected: " << C_ref[index];
          testPassed = false;
        }
        std::cout << ") ";
      } 
    //   else if (std::abs(h_C[index] - C_ref[index]) > 1e-5) {
    //     std::cout << "FAIL at index " << index <<  ", expected: " << C_ref[index] 
    //               << ", got: " << h_C[index] << "\n"; 
    //     testPassed = false;
    //   }
    }
    if (ELEMENT_WISE) 
      std::cout << std::endl;
  }

  std::cout << std::endl;
  if (testPassed) {
    std::cout <<"\t\t\t\t" << kernelName << " Test passed!\n \n";
  } else {
    std::cout <<"\t\t\t\t" << kernelName << " Test failed!\n \n";
  }
}