package com.oddkernel.jamblib;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class Polynomial {

    private List<Double> roots;
    private double amplitude = 1.0;

    public Polynomial(List<Double> roots) {
        this.roots = roots;
    }

    public List<List<Double>> eval(List<List<Double>> x) {
        System.out.println("Roots: " + this.roots);
        List<List<Double>> y = new ArrayList<>();
        for (int indexSample = 0; indexSample < x.size(); indexSample ++) {
            List<Double> sampleResult = x.get(indexSample).stream().map(
                dimension -> (Double) roots.stream().map(root -> dimension - root).reduce((a,b) -> a*b).orElse(0.0)
            ).collect(Collectors.toList());
            y.add(sampleResult);
        }
        return NumJa.divideMatrix(y, this.amplitude);
    }

    public void fit(List<List<Double>> trainFeatures, List<List<Double>> testFeatures) {
        List<List<Double>> trainTagets = eval(trainFeatures);
        Double maxTrain = NumJa.maxMatrix(trainTagets);
        Double minTrain = NumJa.minMatrix(trainTagets);
        List<List<Double>> testTargets = eval(testFeatures);
        Double maxTest = NumJa.maxMatrix(testTargets);
        Double minTest = NumJa.minMatrix(testTargets);
        this.amplitude  = NumJa.max(maxTrain, maxTest) - NumJa.min(minTrain, minTest);
    }
}
