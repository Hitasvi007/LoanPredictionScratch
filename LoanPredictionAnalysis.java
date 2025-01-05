import org.knowm.xchart.*;
import org.knowm.xchart.style.Styler;
import java.awt.Color;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class LoanPredictionAnalysis {

    public static void main(String[] args) {
        List<Map<String, String>> loanData = readCSV("C:\\Users\\Dell\\Documents\\Projects\\Ml\\LoanPredictonAnalysis.csv");

        printFirstNRecords(loanData, 5);
        describeData(loanData);
        dataInfo(loanData);

        loanData = handleMissingValues(loanData);

        countCategories(loanData, "Gender");
        countCategories(loanData, "Married");
        countCategories(loanData, "Dependents");
        countCategories(loanData, "Education");
        countCategories(loanData, "Self_Employed");
        countCategories(loanData, "Property_Area");
        countCategories(loanData, "Loan_Status");

        analyzeNumerical(loanData, "ApplicantIncome");
        analyzeNumerical(loanData, "CoapplicantIncome");
        analyzeNumerical(loanData, "LoanAmount");
        analyzeNumerical(loanData, "Loan_Amount_Term");
        analyzeNumerical(loanData, "Credit_History");

        List<String> numericalColumns = Arrays.asList("ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History");
        calculateAndVisualizeCorrelationMatrix(loanData, numericalColumns);
    }
    
    public static void calculateAndVisualizeCorrelationMatrix(List<Map<String, String>> data, List<String> numericalColumns) {
        int n = numericalColumns.size();
        double[][] correlationMatrix = new double[n][n];

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                correlationMatrix[i][j] = calculateCorrelation(data, numericalColumns.get(i), numericalColumns.get(j));
            }
        }

        System.out.println("Correlation Matrix:");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                System.out.printf("%.2f ", correlationMatrix[i][j]);
            }
            System.out.println();
        }

        XYChart chart = new XYChartBuilder().width(800).height(600).title("Correlation Matrix").xAxisTitle("Features").yAxisTitle("Features").build();
        chart.getStyler().setMarkerSize(10);
        
        List<String> labels = numericalColumns;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                List<Double> xData = Collections.singletonList((double) i);
                List<Double> yData = Collections.singletonList((double) j);
                double correlation = correlationMatrix[i][j];
                Color pointColor = getColorForCorrelation(correlation);
                chart.addSeries(String.format("%s vs %s", labels.get(i), labels.get(j)), xData, yData).setMarkerColor(pointColor);
            }
        }

        new SwingWrapper<>(chart).displayChart();
    }

    public static double calculateCorrelation(List<Map<String, String>> data, String key1, String key2) {
        List<Double> x = new ArrayList<>();
        List<Double> y = new ArrayList<>();

        for (Map<String, String> record : data) {
            if (record.get(key1) != null && record.get(key2) != null) {
                x.add(Double.parseDouble(record.get(key1)));
                y.add(Double.parseDouble(record.get(key2)));
            }
        }

        double meanX = x.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double meanY = y.stream().mapToDouble(Double::doubleValue).average().orElse(0);

        double numerator = 0;
        double denominatorX = 0;
        double denominatorY = 0;

        for (int i = 0; i < x.size(); i++) {
            double diffX = x.get(i) - meanX;
            double diffY = y.get(i) - meanY;

            numerator += diffX * diffY;
            denominatorX += diffX * diffX;
            denominatorY += diffY * diffY;
        }

        double denominator = Math.sqrt(denominatorX) * Math.sqrt(denominatorY);
        return denominator == 0 ? 0 : numerator / denominator;
    }

    public static Color getColorForCorrelation(double correlation) {
        if (correlation > 0.75) return Color.BLUE;
        else if (correlation > 0.5) return Color.CYAN;
        else if (correlation > 0.25) return Color.GREEN;
        else if (correlation > 0) return Color.YELLOW;
        else if (correlation > -0.25) return Color.ORANGE;
        else if (correlation > -0.5) return Color.RED;
        else return Color.MAGENTA;
    }

    public static List<Map<String, String>> readCSV(String fileName) {
        List<Map<String, String>> data = new ArrayList<>();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(fileName));
            String[] headers = reader.readLine().split(",");
            String line;
            while ((line = reader.readLine()) != null) {
                String[] values = line.split(",");
                Map<String, String> record = new HashMap<>();
                for (int i = 0; i < headers.length; i++) {
                    record.put(headers[i], values[i].isEmpty() ? null : values[i]);
                }
                data.add(record);
            }
            reader.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return data;
    }

    public static void printFirstNRecords(List<Map<String, String>> data, int n) {
        for (int i = 0; i < n && i < data.size(); i++) {
            System.out.println(data.get(i));
        }
    }

    // Function to print dataset statistics (simulating .describe())
    public static void describeData(List<Map<String, String>> data) {
        System.out.println("Descriptive statistics...");
    }

    public static void dataInfo(List<Map<String, String>> data) {
        System.out.println("Data Info:");
        if (data.size() > 0) {
            Map<String, String> sample = data.get(0);
            for (String key : sample.keySet()) {
                String value = sample.get(key);
                if (value != null) {
                    System.out.println(key + ": " + value.getClass().getSimpleName());
                } else {
                    System.out.println(key + ": null");
                }
            }
        }
    }

    public static List<Map<String, String>> handleMissingValues(List<Map<String, String>> data) {
        replaceWithMean(data, "LoanAmount");
        replaceWithMean(data, "Loan_Amount_Term");
        replaceWithMean(data, "Credit_History");

        replaceWithMode(data, "Gender");
        replaceWithMode(data, "Married");
        replaceWithMode(data, "Dependents");
        replaceWithMode(data, "Self_Employed");

        return data;
    }

    public static void replaceWithMean(List<Map<String, String>> data, String key) {
        double sum = 0;
        int count = 0;
        for (Map<String, String> record : data) {
            if (record.get(key) != null) {
                sum += Double.parseDouble(record.get(key));
                count++;
            }
        }
        double mean = sum / count;
        for (Map<String, String> record : data) {
            if (record.get(key) == null) {
                record.put(key, String.valueOf(mean));
            }
        }
    }

    public static void replaceWithMode(List<Map<String, String>> data, String key) {
        Map<String, Integer> frequency = new HashMap<>();
        for (Map<String, String> record : data) {
            String value = record.get(key);
            if (value != null) {
                frequency.put(value, frequency.getOrDefault(value, 0) + 1);
            }
        }
        String mode = Collections.max(frequency.entrySet(), Map.Entry.comparingByValue()).getKey();
        for (Map<String, String> record : data) {
            if (record.get(key) == null) {
                record.put(key, mode);
            }
        }
    }

    public static void countCategories(List<Map<String, String>> data, String key) {
        Map<String, Integer> frequency = new HashMap<>();
        for (Map<String, String> record : data) {
            String value = record.get(key);
            frequency.put(value, frequency.getOrDefault(value, 0) + 1);
        }
        System.out.println("Category counts for " + key + ": " + frequency);
        
        // Data visualization using XChart
        CategoryChart chart = new CategoryChartBuilder().width(800).height(600).title("Category Counts for " + key).xAxisTitle(key).yAxisTitle("Count").build();
        chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNW);

        chart.addSeries(key, new ArrayList<>(frequency.keySet()), new ArrayList<>(frequency.values()));
        
        new SwingWrapper<>(chart).displayChart();
    }

    public static void analyzeNumerical(List<Map<String, String>> data, String key) {
        List<Double> values = new ArrayList<>();
        for (Map<String, String> record : data) {
            if (record.get(key) != null) {
                values.add(Double.parseDouble(record.get(key)));
            }
        }
        Collections.sort(values);
        System.out.println("Analysis of " + key + ":");
        System.out.println("Mean: " + values.stream().mapToDouble(Double::doubleValue).average().orElse(0));
        System.out.println("Median: " + values.get(values.size() / 2));
        System.out.println("Min: " + values.get(0));
        System.out.println("Max: " + values.get(values.size() - 1));

        Histogram histogram = new Histogram(values, 20); // Number of bins = 20
        CategoryChart chart = new CategoryChartBuilder().width(800).height(600).title("Histogram for " + key).xAxisTitle(key).yAxisTitle("Frequency").build();
        chart.getStyler().setLegendPosition(Styler.LegendPosition.InsideNW);
        chart.addSeries(key, histogram.getxAxisData(), histogram.getyAxisData());

        new SwingWrapper<>(chart).displayChart();
    }
}
