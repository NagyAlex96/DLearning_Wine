using CNTK;

namespace DLearning_Wine
{
    public enum Activation { ReLU, Sigmoid, None }

    public class Program
    {
        static DeviceDescriptor device = DeviceDescriptor.CPUDevice;

        // Előkészítjük a modell felépítéséhez szükséges változókat
        static int inputDim = 11;     // az adatbázis 11 input feature-t tartalmaz
        static int numOutputClasses = 10; // wine quality értékek 0-10 között

        static void Main(string[] args)
        {
            var trainingData = LoadWineData("winequality.csv");
            Console.WriteLine("Adatok beolvasva");
            var input = Variable.InputVariable(new int[] { inputDim }, DataType.Float);
            var label = Variable.InputVariable(new int[] { numOutputClasses }, DataType.Float);

            var model = CreateNetwork(input, numOutputClasses);
            var trainer = SetupTrainer(model, label);

            Console.WriteLine("Model tanítása");
            // Train the model
            TrainModel(trainer, input, label, trainingData);

            Console.WriteLine("Kiértékelés");
            // Evaluate the model
            EvaluateModel(model, trainingData);

            Console.WriteLine("Sikeres betanítás!!");
            Console.ReadKey();
        }

        static List<Tuple<float[], float[]>> LoadWineData(string path)
        {
            var data = new List<Tuple<float[], float[]>>();
            foreach (var line in File.ReadAllLines(path).Skip(1))
            {
                var values = line.Split(';').Select(x => float.Parse(x, System.Globalization.CultureInfo.InvariantCulture)).ToArray();
                var features = values.Take(inputDim).ToArray();
                var label = new float[numOutputClasses];
                label[(int)values.Last()] = 1.0f;
                data.Add(new Tuple<float[], float[]>(features, label));
            }
            return data;
        }

        // Model létrehozása
        static Function CreateNetwork(Variable input, int outputDim)
        {
            var hiddenLayerDim = 50;
            var layer1 = FullyConnectedLayer(input, hiddenLayerDim, device, Activation.ReLU);
            var outputLayer = FullyConnectedLayer(layer1, outputDim, device, Activation.None);
            return outputLayer;
        }

        static Function FullyConnectedLayer(Variable input, int outputDim, DeviceDescriptor device, Activation activation)
        {
            var inputDim = input.Shape[0];
            var weights = new Parameter(new int[] { outputDim, inputDim }, DataType.Float, CNTKLib.GlorotUniformInitializer(), device);
            var bias = new Parameter(new NDShape(1, outputDim), DataType.Float, 0.0f, device);
            var fullyConnected = CNTKLib.Plus(bias, CNTKLib.Times(weights, input));

            switch (activation)
            {
                case Activation.ReLU:
                    return CNTKLib.ReLU(fullyConnected);
                case Activation.Sigmoid:
                    return CNTKLib.Sigmoid(fullyConnected);
                default:
                    return fullyConnected;
            }
        }

        // Train setup
        static Trainer SetupTrainer(Function model, Variable label)
        {
            var loss = CNTKLib.CrossEntropyWithSoftmax(model, label);
            var evalError = CNTKLib.ClassificationError(model, label);
            var learner = CNTKLib.SGDLearner(new ParameterVector(model.Parameters().ToArray()), new TrainingParameterScheduleDouble(0.01, 1));
            return Trainer.CreateTrainer(model, loss, evalError, new List<Learner> { learner });
        }

        static void TrainModel(Trainer trainer, Variable input, Variable label, List<Tuple<float[], float[]>> trainingData)
        {
            int epochSize = 500;
            int minibatchSize = 64;

            for (int epoch = 0; epoch < epochSize; epoch++)
            {
                var shuffledData = trainingData.OrderBy(_ => Guid.NewGuid()).ToList();
                for (int i = 0; i < shuffledData.Count; i += minibatchSize)
                {
                    var minibatchData = shuffledData.Skip(i).Take(minibatchSize).ToList();
                    var featuresBatch = minibatchData.Select(d => d.Item1).ToArray();
                    var labelsBatch = minibatchData.Select(d => d.Item2).ToArray();

                    // Használjuk a List<float>-et a Value.CreateBatchhoz
                    var inputVal = Value.CreateBatch(input.Shape, featuresBatch.SelectMany(f => f).ToList(), device);
                    var labelVal = Value.CreateBatch(label.Shape, labelsBatch.SelectMany(l => l).ToList(), device);

                    var dataMap = new Dictionary<Variable, Value>
            {
                { input, inputVal },
                { label, labelVal }
            };

                    trainer.TrainMinibatch(dataMap, device);
                }
            }
        }

        static void EvaluateModel(Function model, List<Tuple<float[], float[]>> testData)
        {
            int correct = 0;

            foreach (var sample in testData)
            {
                // Ensure input data is valid
                if (sample.Item1 == null || sample.Item1.Length == 0)
                {
                    Console.WriteLine("Missing input data.");
                    continue; // Skip this iteration if there's no input
                }

                // Create input value batch
                var inputVal = Value.CreateBatch(
                    model.Arguments[0].Shape,
                    sample.Item1.ToList(), // Convert the float array to a List<float>
                    device
                );

                // Create output data map
                var outputDataMap = new Dictionary<Variable, Value>
        {
            { model.Output, null } // Initialize to null for output
        };

                // Evaluate the model
                model.Evaluate(new Dictionary<Variable, Value>
        {
            { model.Arguments[0], inputVal }
        }, outputDataMap, device);

                // Get output data
                var outputData = outputDataMap[model.Output].GetDenseData<float>(model.Output);

                // Predicted label
                var predictedLabel = outputData[0].IndexOf(outputData[0].Max());

                // Actual label
                var actualLabel = Array.IndexOf(sample.Item2, 1.0f);

                // Count correct predictions
                if (predictedLabel == actualLabel) correct++;
            }

            // Output accuracy
            Console.WriteLine($"Accuracy: {correct / (float)testData.Count * 100:F2}%");
        }


    }
}

