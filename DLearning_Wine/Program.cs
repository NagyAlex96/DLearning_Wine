using CNTK;

namespace DLearning_Wine
{
    public enum Activation { ReLU, Sigmoid, None }

    public class Program
    {
        static void Main(string[] args)
        {
            const int epochSize = 1000;        // Növelt epochok száma a pontosabb modellért
            const int minibatchSize = 64;      // Optimalizált minibatch méret
            const int inputDim = 11;           // Bemeneti jellemzők száma
            const int numOutputClasses = 10;   // Kimeneti osztályok száma
            var device = DeviceDescriptor.CPUDevice;

            var dataLoader = new DataLoader();
            var trainingData = dataLoader.LoadWineData("winequality.csv", numOutputClasses);
            Console.WriteLine("Adatok beolvasva");

            var input = Variable.InputVariable(new int[] { inputDim }, DataType.Float);
            var label = Variable.InputVariable(new int[] { numOutputClasses }, DataType.Float);

            var wineModel = new WineModel(input, numOutputClasses, device);
            var modelTrainer = new ModelTrainer(wineModel.Model, label, epochSize, minibatchSize, device);

            Console.WriteLine("Model tanítása");
            modelTrainer.TrainModel(input, label, trainingData);

            Console.WriteLine("Kiértékelés");
            float accuracy = modelTrainer.EvaluateModel(wineModel, trainingData);
            Console.WriteLine($"Pontosság: {accuracy:F2}%");

            Console.WriteLine("Sikeres betanítás!");
            Console.ReadKey();
        }
    }



}

