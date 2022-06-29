package com.oddkernel.jamblib;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class NumJa {

    public static List<Integer> arange(int length) {
        List<Integer> result = new ArrayList<>(length);
        for (int j = 0; j < length; j++) {
            result.add(j);
        }
        return result;
    }

    public static List<Double> multiply(List<Double> vector1, List<Double> vector2) {
        List<Integer> range = arange(vector1.size());
        return range.stream().map(index -> vector1.get(index)*vector2.get(index)).collect(Collectors.toList());
    }
    public static List<Double> multiply(List<Double> vector, double constant) {
        return vector.stream().map(x -> x * constant).collect(Collectors.toList());
    }

    public static List<Double> zeros(Integer length) {
        List<Double> result = new ArrayList<>(length);
        for (int j = 0; j < length; j++) {
            result.add(0.0);
        }
        return result;
    }

    public static Double norm2(List<Double> vector) {
        return vector.stream().map(x->x*x).reduce((a,b)->a+b).orElse(0.0);
    }

    public static List<Double> norm2All(List<List<Double>> vectors) {
        return vectors.stream().map(vector -> norm2(vector)).collect(Collectors.toList());
    }

    public static List<Double> mean(List<List<Double>> data, int axis) {
        List<List<Double>> inputData;
        if (axis == 0) {
            inputData = traspose(data);
        }
        else {
            inputData = data;
        }
        int numberOfDimensions = data.get(0).size();
        List<Double> result = new ArrayList<>(data.get(0).size());
        // For each dimension, compute mean
        for (int indexDimension = 0; indexDimension < numberOfDimensions; indexDimension ++ ) {
            Double mean = mean(inputData.get(0));
            result.add(mean);
        }
        return result;
    }

    private static Double mean(List<Double> vector) {
        return vector.stream().reduce((a,b) -> a+b).orElse(0.0) / vector.size();
    }


    public static List<Double> std(List<List<Double>> data, int axis) {
        List<List<Double>> inputData;
        if (axis == 0) {
            inputData = traspose(data);
        }
        else {
            inputData = data;
        }
        int numberOfDimensions = data.get(0).size();
        List<Double> result = new ArrayList<>(data.get(0).size());
        // For each dimension, compute mean
        for (int indexDimension = 0; indexDimension < numberOfDimensions; indexDimension ++ ) {
            Double std = std(inputData.get(0));
            result.add(std);
        }
        return result;
    }

    private static Double std(List<Double> vector) {
        Double mean = mean(vector);
        return Math.sqrt(vector.stream().map(x -> (x - mean)*(x - mean)).reduce((a,b) -> a+b).orElse(0.0) / (vector.size() - 1.0));
    }

    private static List<List<Double>> traspose(List<List<Double>> data) {
        int size0 = data.size();
        int size1 = data.get(0).size();
        List<List<Double>> result = new ArrayList<>(size1);
        for (int i = 0; i < size1; i++) {
            result.add(new ArrayList<>(size0));
        }

        for (int i = 0; i < size0; i++) {
            for (int j = 0; j < size1; j++) {
                Double value = data.get(i).get(j);
                result.get(j).add(value);
            }
        }
        return result;
    }

    public class OverflowException extends Exception {
    }

    public static List<Boolean> leq(List<Double> vector, double constant) {
        return vector.stream().map(x -> x <= constant).collect(Collectors.toList());
    }

    public static List<Double> minus(List<Double> vector) {
        return vector.stream().map(x -> -x).collect(Collectors.toList());
    }

    public static List<List<Double>> minusAll(List<List<Double>> vector1s, List<List<Double>> vector2s) {
        List<Integer> range = arange(vector1s.size());
        return range.stream().map(index -> minus(vector1s.get(index), vector2s.get(index))).collect(Collectors.toList());
    }

    public static List<Double> minus(List<Double> vector1, List<Double> vector2) {
        List<Integer> range = arange(vector1.size());
        return range.stream().map(index -> vector1.get(index) - vector2.get(index)).collect(Collectors.toList());
    }

    public static List<Double> exp(List<Double> vector) {
        return vector.stream().map(x -> Math.exp(x)).collect(Collectors.toList());
    }

    public static List<Double> sum(List<Double> vector1, List<Double> vector2) {
        List<Integer> range = arange(vector1.size());
        return range.stream().map(index -> vector1.get(index) + vector2.get(index)).collect(Collectors.toList());
    }

    public static List<Double> sum(List<Double> vector, double constant) {
        return vector.stream().map(value -> value + constant).collect(Collectors.toList());
    }

    public static double exp(double x) {
        return Math.exp(x);
    }

    public static List<List<Double>> randomUniform(double low, double high, int shape0, int shape1) {
        double amplitude = high - low;
        List<List<Double>> result = new ArrayList<>(shape0);
        for (int index0 = 0; index0 < shape0; index0++) {
            List<Double> row = new ArrayList<>(shape1);
            for (int index1 = 0; index1 < shape1; index1++) {
                row.add(Math.random() * amplitude + low);
            }
            result.add(row);
        }
        return result;
    }

    public static Double mse(List<List<Double>> predictTargets, List<List<Double>> trueTargets) {
        return mean(norm2All(minusAll(predictTargets, trueTargets)));
    }

    public static Double maxMatrix(List<List<Double>> matrix) {
        return matrix.stream().map(row -> maxVector(row)).reduce((a,b) -> max(a, b)).orElse(0.0);
    }

    public static Double minMatrix(List<List<Double>> matrix) {
        return matrix.stream().map(row -> minVector(row)).reduce((a,b) -> min(a, b)).orElse(0.0);
    }

    public static Double maxVector(List<Double> row) {
        return row.stream().reduce((a,b) -> max(a, b)).orElse(0.0);
    }


    public static Double minVector(List<Double> row) {
        return row.stream().reduce((a,b) -> min(a, b)).orElse(0.0);
    }

    public static Double max(Double a, Double b) {
        return a < b ? b : a;
    }

    public static Double min(Double a, Double b) {
        return -max(-a,-b);
    }

    public static List<List<Double>> divideMatrix(List<List<Double>> matrix, double constant) {
        return matrix.stream().map(vector -> divideVector(vector, constant)).collect(Collectors.toList());
    }
    public static List<Double> divideVector(List<Double> vector, double constant) {
        return vector.stream().map(value -> value / constant).collect(Collectors.toList());
    }
    public static List<Double> cos(List<Double> vector) {
        return vector.stream().map(value -> Math.cos(value)).collect(Collectors.toList());
    }

    public static List<List<Double>> randomLinspaces(double min, double max, int times, int size) {
        return arange(times).stream().map(sampleIndex -> randomLinspace(min, max, size)).collect(Collectors.toList());
    }
    public static List<Double> randomLinspace(double min, double max, int size) {
        System.out.println(min + ":" + max + ":" + size);
        double step  = (max - min) / (size-1);
        System.out.println(step);
        return arange(size).stream().map(index -> (double)index*step + min + 0.5*step*Math.random()).collect(Collectors.toList());
    }

    public static List<Double> sum(List<List<Double>> matrix, int axis) {
        List<List<Double>> inputData;
        if (axis == 0) {
            inputData = traspose(matrix);
        }
        else {
            inputData = matrix;
        }
        int numberOfDimensions = matrix.get(0).size();
        List<Double> result = new ArrayList<>(matrix.get(0).size());
        // For each dimension, compute sun
        for (int indexDimension = 0; indexDimension < numberOfDimensions; indexDimension ++ ) {
            Double sun = sum(inputData.get(0));
            result.add(sun);
        }
        return result;
    }

    private static Double sum(List<Double> list) {
        return list.stream().reduce((a,b) -> a+b).orElse(0.0);
    }

}
