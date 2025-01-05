import java.util.*;
import java.io.*;
import java.util.stream.Collectors;

public class LoanPredictionModel {

    public static void main(String[] args) {
        List<Map<String, String>> loanData = readCSV("C:\\Users\\Dell\\Documents\\Projects\\Ml\\LoanPredictonAnalysis.csv");

        applyLogTransformation(loanData, "ApplicantIncome", "ApplicantIncomeLog");
        applyLogTransformation(loanData, "LoanAmount", "LoanAmountLog");

        String[] colsToDrop = {"ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Total_Income", "Loan_ID"};
        loanData = dropColumns(loanData, colsToDrop);

        String[] categoricalCols = {"Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status", "Dependents"};
        labelEncode(loanData, categoricalCols);

        List<Map<String, String>> X = dropColumns(loanData, new String[]{"Loan_Status"});
        List<String> y = extractColumn(loanData, "Loan_Status");

        Map<String, List<?>> splitData = trainTestSplit(X, y, 0.20);
        List<Map<String, String>> x_train = (List<Map<String, String>>) splitData.get("x_train");
        List<Map<String, String>> x_test = (List<Map<String, String>>) splitData.get("x_test");
        List<String> y_train = (List<String>) splitData.get("y_train");
        List<String> y_test = (List<String>) splitData.get("y_test");

        SVM model = new SVM(x_train.get(0).size()); 
        model.fit(x_train, y_train);
        classify(model, x_train, y_train, x_test, y_test);

        List<String> y_pred = model.predict(x_test);
        int[][] cm = confusionMatrix(y_test, y_pred);

        printConfusionMatrix(cm);

        double precision = calculatePrecision(cm);
        double recall = calculateRecall(cm);
        double f1Score = calculateF1Score(precision, recall);

        System.out.printf("Precision: %.2f%%\n", precision * 100);
        System.out.printf("Recall: %.2f%%\n", recall * 100);
        System.out.printf("F1 Score: %.2f%%\n", f1Score * 100);
    }

    public static class SVM {
        private double[] weights;
        private double bias;
        private double learningRate;
        private double lambda;  
        public SVM(int numFeatures) {
            this.weights = new double[numFeatures];
            this.bias = 0;
            this.learningRate = 0.001;
            this.lambda = 0.01;
        }

        public void fit(List<Map<String, String>> X, List<String> y) {
            int epochs = 1000;
            for (int epoch = 0; epoch < epochs; epoch++) {
                for (int i = 0; i < X.size(); i++) {
                    double[] xi = extractFeatures(X.get(i));
                    int yi = Integer.parseInt(y.get(i)) == 1 ? 1 : -1;  
                    if (yi * (dotProduct(weights, xi) + bias) < 1) {
                        
                        for (int j = 0; j < weights.length; j++) {
                            weights[j] = weights[j] - learningRate * (2 * lambda * weights[j] - yi * xi[j]);
                        }
                        bias = bias + learningRate * yi;
                    } else {
                        
                        for (int j = 0; j < weights.length; j++) {
                            weights[j] = weights[j] - learningRate * (2 * lambda * weights[j]);
                        }
                    }
                }
            }
        }

        public List<String> predict(List<Map<String, String>> X) {
            List<String> predictions = new ArrayList<>();
            for (Map<String, String> xi : X) {
                double prediction = dotProduct(weights, extractFeatures(xi)) + bias;
                predictions.add(prediction >= 0 ? "1" : "0");
            }
            return predictions;
        }

        public double score(List<Map<String, String>> X, List<String> y) {
            List<String> y_pred = predict(X);
            int correct = 0;
            for (int i = 0; i < y.size(); i++) {
                if (y.get(i).equals(y_pred.get(i))) {
                    correct++;
                }
            }
            return (double) correct / y.size();
        }

        private double[] extractFeatures(Map<String, String> data) {
            return data.values().stream()
                .mapToDouble(value -> {
                    try {
                        if (value == null || value.isEmpty()) {
                            return 0.0;  
                        }
                        return Double.parseDouble(value);  
                    } catch (NumberFormatException e) {
                        
                        return 0.0;
                    }
                })
                .toArray();  
        }

        private double dotProduct(double[] w, double[] x) {
            double result = 0.0;
            for (int i = 0; i < w.length; i++) {
                result += w[i] * x[i];
            }
            return result;
        }
    }

    public static void classify(SVM model, List<Map<String, String>> x_train, List<String> y_train,
                                List<Map<String, String>> x_test, List<String> y_test) {
        model.fit(x_train, y_train);
        double accuracy = model.score(x_test, y_test);
        System.out.println("Accuracy: " + (accuracy * 100) + "%");
    }

    public static int[][] confusionMatrix(List<String> y_test, List<String> y_pred) {
        int[][] cm = new int[2][2];  
        for (int i = 0; i < y_test.size(); i++) {
            int actual = Integer.parseInt(y_test.get(i));
            int predicted = Integer.parseInt(y_pred.get(i));
            cm[actual][predicted]++;
        }
        return cm;
    }

    public static void printConfusionMatrix(int[][] cm) {
        System.out.println("Confusion Matrix:");
        for (int[] row : cm) {
            for (int val : row) {
                System.out.print(val + " ");
            }
            System.out.println();
        }
    }

    public static List<Map<String, String>> readCSV(String fileName) {
        List<Map<String, String>> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(fileName))) {
            String line;
            String[] headers = br.readLine().split(",");

            while ((line = br.readLine()) != null) {
                String[] values = line.split(",");
                Map<String, String> row = new HashMap<>();
                for (int i = 0; i < headers.length; i++) {
                    row.put(headers[i], values[i]);
                }
                data.add(row);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }

    public static void applyLogTransformation(List<Map<String, String>> data, String column, String newColumn) {
        for (Map<String, String> row : data) {
            String valueStr = row.get(column);
            try {
                if (valueStr != null && !valueStr.isEmpty()) {
                    double value = Double.parseDouble(valueStr);
                    row.put(newColumn, String.valueOf(Math.log(value)));
                } else {
                    row.put(newColumn, "0.0");  
                }
            } catch (NumberFormatException e) {
                row.put(newColumn, "0.0");  
            }
        }
    }
    public static List<Map<String, String>> dropColumns(List<Map<String, String>> data, String[] columnsToDrop) {
        return data.stream()
                .map(row -> row.entrySet().stream()
                        .filter(e -> !Arrays.asList(columnsToDrop).contains(e.getKey()))
                        .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue)))
                .collect(Collectors.toList());
    }

    public static void labelEncode(List<Map<String, String>> data, String[] categoricalCols) {
        Map<String, Map<String, String>> encoders = new HashMap<>();
        for (String col : categoricalCols) {
            Map<String, String> labelMap = new HashMap<>();
            int label = 0;
            for (Map<String, String> row : data) {
                String value = row.get(col);
                if (!labelMap.containsKey(value)) {
                    labelMap.put(value, String.valueOf(label++));
                }
                row.put(col, labelMap.get(value));
            }
            encoders.put(col, labelMap);
        }
    }

    public static List<String> extractColumn(List<Map<String, String>> data, String column) {
        return data.stream().map(row -> row.get(column)).collect(Collectors.toList());
    }


    public static Map<String, List<?>> trainTestSplit(List<Map<String, String>> X, List<String> y, double testSize) {
        int testLength = (int) (X.size() * testSize);
        List<Map<String, String>> x_train = new ArrayList<>(X.subList(0, X.size() - testLength));
        List<Map<String, String>> x_test = new ArrayList<>(X.subList(X.size() - testLength, X.size()));
        List<String> y_train = new ArrayList<>(y.subList(0, y.size() - testLength));
        List<String> y_test = new ArrayList<>(y.subList(y.size() - testLength, y.size()));

        Map<String, List<?>> result = new HashMap<>();
        result.put("x_train", x_train);
        result.put("x_test", x_test);
        result.put("y_train", y_train);
        result.put("y_test", y_test);
        return result;
    }

    public static double calculatePrecision(int[][] cm) {
        int truePositive = cm[1][1];
        int falsePositive = cm[0][1];
        return (truePositive + falsePositive) == 0 ? 0 : (double) truePositive / (truePositive + falsePositive);
    }

    public static double calculateRecall(int[][] cm) {
        int truePositive = cm[1][1];
        int falseNegative = cm[1][0];
        return (truePositive + falseNegative) == 0 ? 0 : (double) truePositive / (truePositive + falseNegative);
    }

    public static double calculateF1Score(double precision, double recall) {
        return (precision + recall) == 0 ? 0 : 2 * (precision * recall) / (precision + recall);
    }
}
