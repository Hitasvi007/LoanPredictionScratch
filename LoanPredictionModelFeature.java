import java.util.*;
import java.io.*;
import java.util.stream.Collectors;

public class LoanPredictionModelFeature {

    public static void main(String[] args) {
        List<Map<String, String>> loanData = readCSV("C:\\Users\\Dell\\Documents\\Projects\\Ml\\LoanPredictonAnalysis.csv");

        
        applyLogTransformation(loanData, "ApplicantIncome", "ApplicantIncomeLog");
        applyLogTransformation(loanData, "LoanAmount", "LoanAmountLog");
        String[] colsToDrop = {"ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Total_Income", "Loan_ID"};
        loanData = dropColumns(loanData, colsToDrop);
        standardizeFeatures(loanData);
        String[] categoricalCols = {"Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status", "Dependents"};
        labelEncode(loanData, categoricalCols);

        List<String> selectedFeatures = selectTopFeatures(loanData, "Loan_Status", 5);
        loanData = retainSelectedFeatures(loanData, selectedFeatures);

        List<Map<String, String>> X = dropColumns(loanData, new String[]{"Loan_Status"});
        List<String> y = extractColumn(loanData, "Loan_Status");

        int k = 5; 
        crossValidate(X, y, k);

        Map<String, List<?>> splitData = trainTestSplit(X, y, 0.20);
        List<Map<String, String>> x_train = (List<Map<String, String>>) splitData.get("x_train");
        List<Map<String, String>> x_test = (List<Map<String, String>>) splitData.get("x_test");
        List<String> y_train = (List<String>) splitData.get("y_train");
        List<String> y_test = (List<String>) splitData.get("y_test");

        SVM model = new SVM(x_train.get(0).size());
        model.fit(x_train, y_train);
        classify(model, x_train, y_train, x_test, y_test);
        printModelEvaluation(model, x_test, y_test);
        getUserPrediction(model, selectedFeatures); 
    }


    
public static class SVM {
    private double[] weights;
    private double bias;
    private double initialLearningRate;
    private double lambda;  
    private double decay;    

    public SVM(int numFeatures) {
        this.weights = new double[numFeatures];
        this.bias = 0.01;
        this.initialLearningRate = 0.01;
        this.lambda = 0.01;
        this.decay = 0.1; 
    }

    public void fit(List<Map<String, String>> X, List<String> y) {
        int epochs = 1000;
        int batchSize = 32; 

        Map<String, Double> classWeights = calculateClassWeights(y);

        for (int epoch = 0; epoch < epochs; epoch++) {
            List<Integer> indices = new ArrayList<>();
            for (int i = 0; i < X.size(); i++) {
                indices.add(i);
            }
            Collections.shuffle(indices); 

            for (int batchStart = 0; batchStart < X.size(); batchStart += batchSize) {
                int batchEnd = Math.min(batchStart + batchSize, X.size());
                for (int i = batchStart; i < batchEnd; i++) {
                    int index = indices.get(i);
                    double[] xi = extractFeatures(X.get(index));
                    int yi = Integer.parseInt(y.get(index)) == 1 ? 1 : -1;  

                    double classWeight = classWeights.get(y.get(index));

                    if (yi * (dotProduct(weights, xi) + bias) < 1) {
                        for (int j = 0; j < weights.length; j++) {
                            weights[j] -= (initialLearningRate / Math.sqrt(epoch + 1)) * (2 * lambda * weights[j] - yi * classWeight * xi[j]);
                        }
                        bias += (initialLearningRate / Math.sqrt(epoch + 1)) * yi * classWeight; 
                    } else {
                        
                        for (int j = 0; j < weights.length; j++) {
                            weights[j] -= (initialLearningRate / Math.sqrt(epoch + 1)) * (2 * lambda * weights[j]);
                        }
                    }
                }
            }

            initialLearningRate *= (1.0 / (1.0 + decay * epoch));
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

    private Map<String, Double> calculateClassWeights(List<String> y) {
        Map<String, Integer> classCounts = new HashMap<>();
        for (String label : y) {
            classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
        }
        Map<String, Double> classWeights = new HashMap<>();
        double total = y.size();
        for (Map.Entry<String, Integer> entry : classCounts.entrySet()) {
            classWeights.put(entry.getKey(), total / (entry.getValue() * classCounts.size())); 
        }
        return classWeights;
    }

public double decisionFunction(Map<String, String> features) {
    double score = bias; 

    double[] featureValues = extractFeatures(features);
    
    score += dotProduct(weights, featureValues); 

    return score;
}

}
    private static void getUserPrediction(SVM model, List<String> selectedFeatures) {
        Scanner scanner = new Scanner(System.in);
        Map<String, String> userInput = new HashMap<>();

        System.out.println("Enter values for the following features:");
        for (String feature : selectedFeatures) {
            System.out.print(feature + ": ");
            String value = scanner.nextLine();
            userInput.put(feature, value);
        }
        if (userInput.containsKey("ApplicantIncomeLog")) {
            double income = Math.log(Double.parseDouble(userInput.get("ApplicantIncomeLog")) + 1);
            userInput.put("ApplicantIncomeLog", String.valueOf(income));
        }
        if (userInput.containsKey("LoanAmountLog")) {
            double loanAmount = Math.log(Double.parseDouble(userInput.get("LoanAmountLog")) + 1);
            userInput.put("LoanAmountLog", String.valueOf(loanAmount));
        }
        String prediction = model.predict(Collections.singletonList(userInput)).get(0);

        System.out.println("Predicted Loan Status (1 = Approved, 0 = Rejected): " + prediction);
    }

    
    public static void crossValidate(List<Map<String, String>> X, List<String> y, int k) {
        int foldSize = X.size() / k;
        List<Double> accuracies = new ArrayList<>();

        for (int i = 0; i < k; i++) {
            List<Map<String, String>> x_train = new ArrayList<>();
            List<String> y_train = new ArrayList<>();
            List<Map<String, String>> x_test = new ArrayList<>();
            List<String> y_test = new ArrayList<>();

            for (int j = 0; j < X.size(); j++) {
                if (j >= i * foldSize && j < (i + 1) * foldSize) {
                    x_test.add(X.get(j));
                    y_test.add(y.get(j));
                } else {
                    x_train.add(X.get(j));
                    y_train.add(y.get(j));
                }
            }
            SVM model = new SVM(x_train.get(0).size());
            model.fit(x_train, y_train);

            double accuracy = model.score(x_test, y_test);
            accuracies.add(accuracy);
            System.out.printf("Fold %d - Accuracy: %.2f%%\n", i + 1, accuracy * 100);
        }

        double averageAccuracy = accuracies.stream().mapToDouble(a -> a).average().orElse(0.0);
        System.out.printf("Average Accuracy after %d-fold cross-validation: %.2f%%\n", k, averageAccuracy * 100);
    }

   
public static void standardizeFeatures(List<Map<String, String>> data) {
    Map<String, Double> means = new HashMap<>();
    Map<String, Double> stdDevs = new HashMap<>();

    for (String feature : data.get(0).keySet()) {
        if (isNumericFeature(data, feature)) {
            List<Double> values = data.stream()
                .map(row -> {
                    String value = row.get(feature);
                    return (value == null || value.isEmpty()) ? 0.0 : Double.parseDouble(value);
                })
                .collect(Collectors.toList());

            double mean = values.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
            means.put(feature, mean);

            double stdDev = Math.sqrt(values.stream()
                .mapToDouble(val -> Math.pow(val - mean, 2)).sum() / values.size());
            stdDevs.put(feature, stdDev);
        }
    }
    for (Map<String, String> row : data) {
        for (String feature : row.keySet()) {
            if (means.containsKey(feature)) {
                double originalValue = (row.get(feature) == null || row.get(feature).isEmpty()) ? 0.0 : Double.parseDouble(row.get(feature));
                double standardizedValue = (originalValue - means.get(feature)) / stdDevs.get(feature);
                row.put(feature, String.valueOf(standardizedValue));
            }
        }
    }
}

private static boolean isNumericFeature(List<Map<String, String>> data, String feature) {
    for (Map<String, String> row : data) {
        String value = row.get(feature);
        if (value != null && !value.isEmpty()) {
            try {
                Double.parseDouble(value); 
            } catch (NumberFormatException e) {
                return false; 
            }
        }
    }
    return true; 
}
public static Map<String, Double> calculateClassWeights(List<String> y) {
    Map<String, Integer> classCounts = new HashMap<>();
    for (String label : y) {
        classCounts.put(label, classCounts.getOrDefault(label, 0) + 1);
    }

    Map<String, Double> classWeights = new HashMap<>();
    double totalSamples = y.size();

    for (Map.Entry<String, Integer> entry : classCounts.entrySet()) {
        double weight = totalSamples / (2 * entry.getValue()); 
        classWeights.put(entry.getKey(), weight);
    }

    return classWeights;
}


    public static List<Map<String, String>> readCSV(String filePath) {
        List<Map<String, String>> data = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line = br.readLine();
            if (line == null) return data;
            
            String[] headers = line.split(",");
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
    
    public static void applyLogTransformation(List<Map<String, String>> data, String colName, String newColName) {
        for (Map<String, String> row : data) {
            String value = row.get(colName);
            if (value != null && !value.isEmpty()) {
                double originalValue = Double.parseDouble(value);
                row.put(newColName, String.valueOf(Math.log1p(originalValue)));
            } else {
                row.put(newColName, "0");
            }
        }
    }
    
    public static List<Map<String, String>> dropColumns(List<Map<String, String>> data, String[] colsToDrop) {
        List<Map<String, String>> modifiedData = new ArrayList<>();
        for (Map<String, String> row : data) {
            Map<String, String> newRow = new HashMap<>(row);
            for (String col : colsToDrop) {
                newRow.remove(col);
            }
            modifiedData.add(newRow);
        }
        return modifiedData;
    }
    
    public static void labelEncode(List<Map<String, String>> data, String[] categoricalCols) {
        Map<String, Map<String, String>> encoders = new HashMap<>();
        
        for (String col : categoricalCols) {
            Map<String, String> encoder = new HashMap<>();
            int label = 0;
            for (Map<String, String> row : data) {
                String value = row.get(col);
                if (value != null && !encoder.containsKey(value)) {
                    encoder.put(value, String.valueOf(label++));
                }
                row.put(col, encoder.get(value));
            }
            encoders.put(col, encoder);
        }
    }
    
    public static List<String> extractColumn(List<Map<String, String>> data, String colName) {
        return data.stream()
                   .map(row -> row.getOrDefault(colName, "0"))
                   .collect(Collectors.toList());
    }
    
    public static Map<String, List<?>> trainTestSplit(List<Map<String, String>> X, List<String> y, double testSize) {
        int testSizeCount = (int) (X.size() * testSize);
        List<Map<String, String>> x_train = new ArrayList<>(X.subList(0, X.size() - testSizeCount));
        List<Map<String, String>> x_test = new ArrayList<>(X.subList(X.size() - testSizeCount, X.size()));
        List<String> y_train = new ArrayList<>(y.subList(0, y.size() - testSizeCount));
        List<String> y_test = new ArrayList<>(y.subList(y.size() - testSizeCount, y.size()));
    
        Map<String, List<?>> splitData = new HashMap<>();
        splitData.put("x_train", x_train);
        splitData.put("x_test", x_test);
        splitData.put("y_train", y_train);
        splitData.put("y_test", y_test);
    
        return splitData;
    }
    
    public static void printModelEvaluation(SVM model, List<Map<String, String>> x_test, List<String> y_test) {
        List<String> y_pred = model.predict(x_test);

        int[][] cm = confusionMatrix(y_test, y_pred);
        System.out.println("Confusion Matrix:");
        printConfusionMatrix(cm);

        double precision = calculatePrecision(cm);
        double recall = calculateRecall(cm);
        double f1Score = calculateF1Score(precision, recall);
        double accuracy = model.score(x_test, y_test);

        System.out.printf("Precision: %.2f%%\n", precision * 100);
        System.out.printf("Recall: %.2f%%\n", recall * 100);
        System.out.printf("F1 Score: %.2f%%\n", f1Score * 100);
        System.out.printf("Accuracy: %.2f%%\n", accuracy * 100);
    }

    
    public static List<String> selectTopFeatures(List<Map<String, String>> data, String targetColumn, int topN) {
        Map<String, Double> featureCorrelation = new HashMap<>();

        for (String feature : data.get(0).keySet()) {
            if (!feature.equals(targetColumn)) {
                double correlation = calculateCorrelation(data, feature, targetColumn);
                featureCorrelation.put(feature, Math.abs(correlation));  
            }
        }

        return featureCorrelation.entrySet().stream()
            .sorted((a, b) -> b.getValue().compareTo(a.getValue())) 
            .limit(topN)
            .map(Map.Entry::getKey)
            .collect(Collectors.toList());
    }

    public static double calculateCorrelation(List<Map<String, String>> data, String feature, String target) {
        List<Double> featureValues = data.stream()
            .map(row -> {
                String value = row.get(feature);
                return (value == null || value.isEmpty()) ? 0.0 : Double.parseDouble(value);  
            })
            .collect(Collectors.toList());
    
        List<Double> targetValues = data.stream()
            .map(row -> {
                String value = row.get(target);
                return (value == null || value.isEmpty()) ? 0.0 : Double.parseDouble(value);  
            })
            .collect(Collectors.toList());
    
        return pearsonCorrelation(featureValues, targetValues);
    }
    

    public static double pearsonCorrelation(List<Double> x, List<Double> y) {
        int n = x.size();
        double sumX = x.stream().mapToDouble(Double::doubleValue).sum();
        double sumY = y.stream().mapToDouble(Double::doubleValue).sum();
        double sumXY = 0, sumX2 = 0, sumY2 = 0;

        for (int i = 0; i < n; i++) {
            sumXY += x.get(i) * y.get(i);
            sumX2 += x.get(i) * x.get(i);
            sumY2 += y.get(i) * y.get(i);
        }

        return (n * sumXY - sumX * sumY) / (Math.sqrt(n * sumX2 - sumX * sumX) * Math.sqrt(n * sumY2 - sumY * sumY));
    }

    public static List<Map<String, String>> retainSelectedFeatures(List<Map<String, String>> data, List<String> selectedFeatures) {
        return data.stream()
            .map(row -> row.entrySet().stream()
                    .filter(e -> selectedFeatures.contains(e.getKey()) || e.getKey().equals("Loan_Status"))
                    .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue)))
            .collect(Collectors.toList());
    }

   

    public static void classify(SVM model, List<Map<String, String>> x_train, List<String> y_train, List<Map<String, String>> x_test, List<String> y_test) {
        double trainAccuracy = model.score(x_train, y_train);
        double testAccuracy = model.score(x_test, y_test);

        System.out.printf("Training Accuracy: %.2f%%\n", trainAccuracy * 100);
        System.out.printf("Testing Accuracy: %.2f%%\n", testAccuracy * 100);
    }

    public static int[][] confusionMatrix(List<String> y_true, List<String> y_pred) {
        int[][] cm = new int[2][2];
        for (int i = 0; i < y_true.size(); i++) {
            int actual = Integer.parseInt(y_true.get(i));
            int predicted = Integer.parseInt(y_pred.get(i));
            cm[actual][predicted]++;
        }
        return cm;
    }

    public static double calculatePrecision(int[][] cm) {
        return (double) cm[1][1] / (cm[1][1] + cm[0][1]);
    }

    public static double calculateRecall(int[][] cm) {
        return (double) cm[1][1] / (cm[1][1] + cm[1][0]);
    }

    public static double calculateF1Score(double precision, double recall) {
        return 2 * (precision * recall) / (precision + recall);
    }

    public static void printConfusionMatrix(int[][] cm) {
        System.out.println("True Negative: " + cm[0][0]);
        System.out.println("False Positive: " + cm[0][1]);
        System.out.println("False Negative: " + cm[1][0]);
        System.out.println("True Positive: " + cm[1][1]);
    }
}

