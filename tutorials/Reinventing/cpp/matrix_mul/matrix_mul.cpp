#include <iostream>

void initialize_matrix(int *data, int row, int col){
    for(int r=0; r<row; r++){
        for (int c=0; c<col;c++)
            data[r*col+c] = r+c;
    }
}

void display_matrix(int *data, int row, int col){
    for(int r=0; r<row; r++){
        for(int c=0;c<col; c++)
            std::cout<<data[r*col+c]<<" ";
        std::cout<<std::endl;
    }
}


void matrix_multiply(int *A, int *B, int *C, int row_a, int col_a, int col_b){

    // Memory Access is Reduced by Loop-Reordering and Reducing Memory Access
    for(int k=0; k<col_a; k++){
        for(int r=0; r<row_a; r++){
            auto temp = A[r*col_a+k];
            for(int c=0; c<col_b; c++)
                C[r*col_b+c]+=temp*B[k*col_b+c];
        }
    }
}

void naive_matrix_multiply(int *A, int *B, int *C, int row_a, int col_a, int col_b){
    
    for(int k=0; k<col_a;k++){
        for(int r=0; r<row_a; r++){
        for(int c=0;c<col_b;c++)               
             C[r*col_b+c]+=A[r*col_a+k]*B[k*col_b+c];
        }
    }

}

int main(){
    int a_row = 2, a_col = 3, b_row = 3 , b_col = 2;
    int c_row = a_row, c_col = b_col;
    int a[a_row][a_col], b[b_row][b_col], c[c_row][c_col];

    initialize_matrix(&a[0][0], a_row, a_col);
    initialize_matrix(&b[0][0], b_row, b_col);
    std::cout<<"------------- A -------------"<<std::endl;
    display_matrix(&a[0][0], a_row, a_col);
    std::cout<<"------------- B -------------"<<std::endl;
    display_matrix(&b[0][0], b_row, b_col);
    matrix_multiply(&a[0][0], &b[0][0], &c[0][0], a_row, a_col, b_col);
    std::cout<<"------------- C -------------"<<std::endl;
    display_matrix(&c[0][0], c_row, c_col);
    return 0;

}