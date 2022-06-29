package com.oddkernel.jamblib;

import java.util.List;
import java.util.stream.Collectors;

public class RandomWave extends Polynomial {

    private int nWabes;
    private List<Double> amplitudes;
    private List<Double> frequencies;

    public RandomWave(int nWaves) {
        super(null);
        this.nWabes = nWaves;
        this.amplitudes = NumJa.randomLinspace(0.5, 2., nWaves);
        this.frequencies = NumJa.randomLinspace(0.5, 2., nWaves);
    }

    public List<List<Double>> eval(List<List<Double>> x) {
        System.out.println(amplitudes);
        System.out.println(frequencies);
        return NumJa.arange(x.size()).stream().map(index -> NumJa.arange(x.size()).stream().map(indexF -> eval(x, index, indexF)).reduce((f1, f2) -> NumJa.sum(f1, f2)).orElse(null)).collect(Collectors.toList());
    }

    private List<Double> eval(List<List<Double>> x, int index, int indexF) {
        List<Double> f = NumJa.multiply(NumJa.multiply(x.get(index), this.frequencies.get(indexF)), 2*Math.PI);
        List<Double> cosines = NumJa.cos(f);
        System.out.println(f);
        System.out.println(cosines);
        return NumJa.multiply(cosines, this.amplitudes.get(indexF));

    }
}
