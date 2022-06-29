package com.oddkernel.jamblib;

import java.util.List;

public class MLPRegressor extends MLPerceptron {

    public MLPRegressor(String name, int featureDimension, int targetDimension, List<Integer> hiddenLayerSizes) {
        super(name, featureDimension, targetDimension, hiddenLayerSizes);
    }
    @Override
    public Double finalTransferFunction(Double x) {
        return x;
    }

    @Override
    public Double dFinalTransferFunction(Double x) {
        return 1.0;
    }

}
