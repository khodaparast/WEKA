import weka.classifiers.trees.J48;
import weka.core.Instance;
import weka.core.Instances;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

public class DecisionTree {
    private Instances trainingData;
    public DecisionTree(String fileName) throws IOException {
        Utils utils=new Utils();
        trainingData=utils.getInstances(utils.getReader(fileName));
        trainingData.setClassIndex(trainingData.numAttributes()-1);

    }

    private J48 performTraining() {
        J48 j48 = new J48();
        String[] options = {"-U"};
        try {
            j48.setOptions(options);
            j48.buildClassifier(trainingData);
        } catch (Exception ex) {
            // 
        }
        return j48;
    }

    private Instance getTestInstance(String binding, String multiColor,String genre) {
        Instance instance=new Instance(3);
        instance.setDataset(trainingData);
        instance.setValue(trainingData.attribute(0), binding);
        instance.setValue(trainingData.attribute(1), multiColor);
        instance.setValue(trainingData.attribute(2), genre);

        return instance;
    }
    public static void main(String[] args) throws Exception {
        String fileName="D:/books.arff";
        DecisionTree decisionTree=new DecisionTree(fileName);
        J48 tree=decisionTree.performTraining();
        System.out.println(tree.toString());
        Instance testInstance=decisionTree.getTestInstance("Paperbank","yes","historical"); 
        int result= (int) tree.classifyInstance(testInstance);
        System.out.println(decisionTree.trainingData.attribute(3).value(result));
        System.out.println(tree.classifyInstance());

    }

}
