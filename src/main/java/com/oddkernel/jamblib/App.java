package com.oddkernel.jamblib;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.jfree.chart.ChartUtils;

/**
 * Hello world!
 *
 */
// mvn clean install && java -cp libs/org/jfree/jfreechart/1.5.3/jfreechart-1.5.3.jar:target/jamlib-1.0-SNAPSHOT.jar com.oddkernel.jamblib.App 2,3,3,2 0.01 100 True 0.5

public class App 
{
    public static int HIDDEN_LAYER_SIZES_ARG = 0;
    public static int LEARNING_RATE_ARG = 1;
    public static int EPOCHS_ARG = 2;
    public static int NORMALIZE_ARG = 3;
    public static int TRAIN_PROPORTION_ARG = 4;
    public static int N_ARGS = 5;
    public static void main( String[] args )
    {
        if (args.length < N_ARGS) {
            System.out.println("App <layer_sizes> <learning_rate> <epochs> <normalize> <train_proportion>");
            return;
        }
        List<String> rawHiddenLayerSizes = Arrays.asList(args[HIDDEN_LAYER_SIZES_ARG].split(","));
        List<Integer> hiddenLayerSizes = rawHiddenLayerSizes.stream().map(rawSize -> Integer.parseInt(rawSize)).collect(Collectors.toList());
        Double learningRate = Double.parseDouble(args[LEARNING_RATE_ARG]);
        Integer epochs = Integer.parseInt(args[EPOCHS_ARG]);
        Boolean normalize = Boolean.parseBoolean(args[NORMALIZE_ARG]);
        Double trainProportion = Double.parseDouble(args[TRAIN_PROPORTION_ARG]);

        double alpha = 0.01;
        int totalSize = 1000;
        int trainSize = (int) (totalSize * trainProportion);
        int testSize = (int) (totalSize * (1.0 - trainProportion));
        double x_min = -1.0;
        double x_max = 1.0;
        int nRoots = 5;
        int nWaves = 5;
        double rootsPadding = 1.1;
    
        MLPerceptron model = new MLPRegressor("MLPRegressor", 1, 1, hiddenLayerSizes);
        model.randomizeSynapses(-1., 1.);
        System.out.println("Randomized");
        System.out.println(model);

        List<List<Double>> trainFeatures = NumJa.randomUniform(x_min, x_max, trainSize, 1);
        List<List<Double>> testFeatures = NumJa.randomUniform(x_min, x_max, testSize, 1);

        System.out.println("Data");

        List<List<Double>> roots = NumJa.randomLinspaces(rootsPadding*x_min, rootsPadding*x_max, 1, nRoots);
        
        Polynomial polynomial = new Polynomial(roots.get(0));
        
        polynomial.fit(trainFeatures, testFeatures);

        List<List<Double>> trainTargets = polynomial.eval(trainFeatures);
        List<List<Double>> testTargets = polynomial.eval(testFeatures);
        
        /*
         * 
        RandomWave wave = new RandomWave(nWaves);
        
        wave.fit(trainFeatures, testFeatures);

        List<List<Double>> trainTargets = wave.eval(trainFeatures);
        List<List<Double>> testTargets = wave.eval(testFeatures);
         */

        ScatterPlot plot = new ScatterPlot("Function sample", trainFeatures, trainTargets, null);
        /**
         * 
        SwingUtilities.invokeLater(() -> {  
            plot.setSize(800, 400);  
            plot.setLocationRelativeTo(null);  
            plot.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);  
            plot.setVisible(true);  
          });
         */
        try {
            ChartUtils.saveChartAsPNG(new File("./img/poly.png"), plot.getChart(), 450, 400);
        }
        catch (IOException io) {
            System.out.println("ERROR SAVING PICTURE");
        }

        System.out.println("Starting train");
        
        long start = System.currentTimeMillis();
        model.train(trainFeatures, trainTargets, learningRate, epochs, normalize);
        long finish = System.currentTimeMillis();
        long timeElapsed = finish - start;
        System.out.println(String.format("Elapsed time: %.2f seconds", timeElapsed / 1000.0));
    
        ModelErrorStats trainStats = model.computeStats(trainFeatures, trainTargets);
    
        System.out.println(String.format("Train Score: %.4f", trainStats.getScore()));
        System.out.println("Confussion Matrix:");
        System.out.println(prettyMatrix(trainStats.getConfussionMatrix()));

        ModelErrorStats testStats = model.computeStats(testFeatures, testTargets);
    
        System.out.println(String.format("Train Score: %.4f", testStats.getScore()));
        System.out.println("Confussion Matrix:");
        System.out.println(prettyMatrix(testStats.getConfussionMatrix()));
    }

	public static String prettyMatrix(List<List<Double>> matrix) {
        String result = "";
        if (matrix == null) {
            return result;
        }
        for (List<Double> row : matrix) {
            for (Double element : row) {
                result += String.valueOf(element) + "\t";
            }
            result += "\n";
        }
        return result;
    }
}
/**
 * 


def main():
    """Main function.
    """

    parser = Parser()

    args = vars(parser.parse_args())

    sizes = [int(s) for s in args['sizes']]
    init = float(args['init'])
    learn = float(args['learn'])
    epochs = int(args['epochs'])
    normalize = args['normalize']

    if args['mode'] == 'mode1':
        sizein, sizeout, train, test = mode1(args['data'], args['ratio'])
    elif args['mode'] == 'mode2':
        sizein, sizeout, train, test = mode2(args['data'])
    elif args['mode'] == 'mode3':
        sizein, sizeout, train, test = mode3(args['train'], args['test'])

    p = MLPerceptron('MLPerceptron', sizein, sizeout, sizes)
    p.randomize_synapses(-init, init)

    System.out.println()
    t0 = time()
    p.train(train[0], train[1], learn, epochs, normalize=normalize)
    t = time()
    System.out.println()
    System.out.println('Elapsed time: {0:.3f} seconds'.format(t - t0))
    System.out.println()

    score, m = p.stats(train[0], train[1])

    System.out.println('Train Score:', score)

    if args['mode'] == 'mode1':
        System.out.println('Test Score:', p.stats(test[0], test[1])[0])
    elif args['mode'] == 'mode2':
        System.out.println('Test Score:', score)
    elif args['mode'] == 'mode3':
        f = args['output']
        res = p.test(test[0])
        with open(f, 'w') as fileout:
            fileout.write('{} {}\n'.format(sizein, sizeout))
            for i in range(len(res)):
                x = ' '.join([str(x) for x in test[0][i]])
                r = [-1] * len(res[i])
                r[res[i].index(max(res[i]))] = 1
                y = ' '.join([str(y) for y in r])
                fileout.write('{} {}\n'.format(x, y))
        System.out.println('Test predictions were written to \'{}\'.'.format(f))

    System.out.println('Confussion Matrix:')
    prettymatrix(m)
 */
