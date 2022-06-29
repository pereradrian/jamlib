package com.oddkernel.jamblib;

import java.util.List;

public class ModelErrorStats {

    private Double mse;

    public ModelErrorStats(Double mse) {
        this.mse = mse;
    }

    public Double getScore() {
        return mse;
    }

    public List<List<Double>> getConfussionMatrix() {
        return null;
    }

}
