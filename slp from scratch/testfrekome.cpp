#include <bits/stdc++.h>
using namespace std;

struct Data {
  int id;
  double sepal_length, sepal_width, petal_length, petal_width, species;
};

double amplitudo = 1;
double frekuensi = 2.5;
int main (){
  string filename = "Iris.csv";
  ifstream file(filename);
  string line;

  if (!file.is_open()) {
      cout << "Error: Gagal membuka file " << filename << endl;
      return 1;
  }

  cout << "File " << filename << " berhasil dibuka." << endl;

  // Lewati baris pertama (Header CSV: Id,SepalLengthCm,...)
  getline(file, line);

  vector<Data> dataset;

  while (getline(file, line)){
    stringstream ss(line);
    string cell;
    vector<string> row;

    while (getline(ss, cell, ',')) {
      row.push_back(cell);
    }

    if (row.size() == 6) {
      Data data;
      data.id = stoi(row[0]);
      data.sepal_length = stod(row[1]);
      data.sepal_width = stod(row[2]);
      data.petal_length = stod(row[3]);
      data.petal_width = stod(row[4]);
      data.species = stod(row[5]);
      dataset.push_back(data);
    }
  }
  file.close();

  // for(auto &data : dataset) {
  //   cout << "ID: " << data.id
  //        << ", Sepal Length: " << data.sepal_length
  //        << ", Sepal Width: " << data.sepal_width
  //        << ", Petal Length: " << data.petal_length
  //        << ", Petal Width: " << data.petal_width
  //        << ", Species: " << data.species << endl;
  // } 
  double teta1, teta2, teta3, teta4, bias, learning_rate;
  learning_rate = 0.01;
  teta1 = teta2 = teta3 = teta4 = bias = 0.5;
  vector<double> mse_values;
  for(int epoch = 1; epoch <= 5000; epoch++) {
    double total_square_error = 0.0;
    for(int i = 0; i < dataset.size(); i++) {
      double z = (teta1 * dataset[i].sepal_length) + (teta2 * dataset[i].sepal_width) + (teta3 * dataset[i].petal_length) + (teta4 * dataset[i].petal_width) + bias;
      double sigmoid_val = 1.0 / (1.0 + exp(-frekuensi * z));
      double prediksi = amplitudo * sigmoid_val;
      double error = prediksi - dataset[i].species;
      double square_error = error * error;
      double g_prime = amplitudo * frekuensi * sigmoid_val * (1.0 - sigmoid_val);
      double delta = 2.0 * error * g_prime;

      // Update weights and bias
      teta1 -= learning_rate * delta * dataset[i].sepal_length;
      teta2 -= learning_rate * delta * dataset[i].sepal_width;
      teta3 -= learning_rate * delta * dataset[i].petal_length;
      teta4 -= learning_rate * delta * dataset[i].petal_width;
      bias -= learning_rate * delta;

      // if(epoch == 1 && i == 0){
      //   cout << "Epoch: " << epoch << ", Data Index: " << i << endl;
      //   cout << "Prediksi: " << prediksi << ", Error: " << error << ", Square Error: " << square_error << endl;
      //   cout << "Updated Weights and Bias:" << endl;
      //   cout << "teta1: " << teta1 << ", teta2: " << teta2
      //        << ", teta3: " << teta3 << ", teta4: " << teta4
      //        << ", bias: " << bias << endl;
      // }

      total_square_error += square_error;
    }
    double mse = total_square_error / dataset.size();
    mse_values.push_back(mse);
  }

  for(int epoch = 0; epoch < mse_values.size(); epoch++) {
    if(epoch == 0 || epoch % 100 == 99) {
      cout << "Epoch: " << (epoch + 1) << ", MSE: " << mse_values[epoch] << endl;
    }
  }
}