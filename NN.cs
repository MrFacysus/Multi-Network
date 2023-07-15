		class NeuralNetwork
		{
			private const double learningRate = 0.00015f;

			private List<List<double>> biases;
			private List<List<double>> errors;
			private List<List<double>> values;
			private List<List<List<double>>> weights;

			private Random r = new Random();

			public NeuralNetwork(int[] layers)
			{
				initNeurons(layers);
				initWeights(layers);
			}

			private void initNeurons(int[] layers)
			{
				biases = new List<List<double>>();
				errors = new List<List<double>>();
				values = new List<List<double>>();

				for (int layerIdx = 0; layerIdx < layers.Length; layerIdx++)
				{
					biases.Add(new List<double>());
					errors.Add(new List<double>());
					values.Add(new List<double>());

					for (int neuronIdx = 0; neuronIdx < layers[layerIdx]; neuronIdx++)
					{
						biases[layerIdx].Add(0);
						errors[layerIdx].Add(0);
						values[layerIdx].Add(0);
					}
				}
			}

			private void initWeights(int[] layers)
			{
				weights = new List<List<List<double>>>();

				for (int layerIdx = 0; layerIdx < layers.Length - 1; layerIdx++)
				{
					weights.Add(new List<List<double>>());

					for (int neuronIdx = 0; neuronIdx < layers[layerIdx]; neuronIdx++)
					{
						weights[layerIdx].Add(new List<double>());

						for (int nextNeuronIdx = 0; nextNeuronIdx < layers[layerIdx + 1]; nextNeuronIdx++)
						{
							weights[layerIdx][neuronIdx].Add(r.NextDouble() - 0.5f);
						}
					}
				}
			}

			private void activateTanh()
			{
				for (int layerIdx = 1; layerIdx < values.Count(); layerIdx++)
				{
					for (int neuronIdx = 0; neuronIdx < values[layerIdx].Count(); neuronIdx++)
					{
						double sum = 0;
						for (int prevNeuronIdx = 0; prevNeuronIdx < values[layerIdx - 1].Count(); prevNeuronIdx++)
						{
							sum += values[layerIdx - 1][prevNeuronIdx] * weights[layerIdx - 1][prevNeuronIdx][neuronIdx];
						}
						values[layerIdx][neuronIdx] = Math.Tanh(sum + biases[layerIdx][neuronIdx]);
					}
				}
			}

			private void activateReLU()
			{
				for (int layerIdx = 1; layerIdx < values.Count(); layerIdx++)
				{
					for (int neuronIdx = 0; neuronIdx < values[layerIdx].Count(); neuronIdx++)
					{
						double sum = 0;
						for (int prevNeuronIdx = 0; prevNeuronIdx < values[layerIdx - 1].Count(); prevNeuronIdx++)
						{
							sum += values[layerIdx - 1][prevNeuronIdx] * weights[layerIdx - 1][prevNeuronIdx][neuronIdx];
						}
						values[layerIdx][neuronIdx] = Math.Max(0, sum + biases[layerIdx][neuronIdx]);
					}
				}
			}

			private void activateSigmoid()
			{
				for (int layerIdx = 1; layerIdx < values.Count(); layerIdx++)
				{
					for (int neuronIdx = 0; neuronIdx < values[layerIdx].Count(); neuronIdx++)
					{
						double sum = 0;
						for (int prevNeuronIdx = 0; prevNeuronIdx < values[layerIdx - 1].Count(); prevNeuronIdx++)
						{
							sum += values[layerIdx - 1][prevNeuronIdx] * weights[layerIdx - 1][prevNeuronIdx][neuronIdx];
						}
						values[layerIdx][neuronIdx] = 1 / (1 + Math.Exp(-(sum + biases[layerIdx][neuronIdx])));
					}
				}
			}

			public double[] getTanhOutput(double[] inputs)
			{
                System.Diagnostics.Contracts.Contract.Requires(inputs.Length == values[0].Count());

				for (int neuronIdx = 0; neuronIdx < inputs.Length; neuronIdx++)
				{
					values[0][neuronIdx] = inputs[neuronIdx];
				}

				activateTanh();

				return values[values.Count() - 1].ToArray();
			}

			public double[] getReLUOutput(double[] inputs)
			{
				System.Diagnostics.Contracts.Contract.Requires(inputs.Length == values[0].Count());

				for (int neuronIdx = 0; neuronIdx < inputs.Length; neuronIdx++)
				{
					values[0][neuronIdx] = inputs[neuronIdx];
				}

				activateReLU();

				return values[values.Count() - 1].ToArray();
			}

			public double[] getSigmoidOutput(double[] inputs)
			{
				System.Diagnostics.Contracts.Contract.Requires(inputs.Length == values[0].Count());

				for (int neuronIdx = 0; neuronIdx < inputs.Length; neuronIdx++)
				{
					values[0][neuronIdx] = inputs[neuronIdx];
				}

				activateSigmoid();

				return values[values.Count() - 1].ToArray();
			}

			public void backPropagateTanh(double[] correctOutput)
			{
				for (int neuronIdx = 0; neuronIdx < values[values.Count() - 1].Count(); neuronIdx++)
				{
					errors[errors.Count() - 1][neuronIdx] = (correctOutput[neuronIdx] - values[values.Count() - 1][neuronIdx]) * (1 - values[values.Count() - 1][neuronIdx] * values[values.Count() - 1][neuronIdx]);
				}

				for (int layerIdx = values.Count() - 2; layerIdx > 0; layerIdx--)
				{
					for (int neuronIdx = 0; neuronIdx < values[layerIdx].Count(); neuronIdx++)
					{
						double error = 0;
						for (int nextNeuronIdx = 0; nextNeuronIdx < values[layerIdx + 1].Count(); nextNeuronIdx++)
						{
							error += errors[layerIdx + 1][nextNeuronIdx] * weights[layerIdx][neuronIdx][nextNeuronIdx];
						}
						errors[layerIdx][neuronIdx] = error * (1 - values[layerIdx][neuronIdx] * values[layerIdx][neuronIdx]);
					}
				}

				for (int layerIdx = values.Count() - 1; layerIdx > 0; layerIdx--)
				{
					for (int neuronIdx = 0; neuronIdx < values[layerIdx].Count(); neuronIdx++)
					{
						biases[layerIdx][neuronIdx] += errors[layerIdx][neuronIdx] * learningRate;
						for (int prevNeuronIdx = 0; prevNeuronIdx < values[layerIdx - 1].Count(); prevNeuronIdx++)
						{
							weights[layerIdx - 1][prevNeuronIdx][neuronIdx] += values[layerIdx - 1][prevNeuronIdx] * errors[layerIdx][neuronIdx] * learningRate;
						}
					}
				}
			}

			public void backPropagateReLU(double[] correctOutput)
			{
				for (int neuronIdx = 0; neuronIdx < values[values.Count() - 1].Count(); neuronIdx++)
				{
					errors[errors.Count() - 1][neuronIdx] = (correctOutput[neuronIdx] - values[values.Count() - 1][neuronIdx]) * (values[values.Count() - 1][neuronIdx] > 0 ? 1 : 0);
				}

				for (int layerIdx = values.Count() - 2; layerIdx > 0; layerIdx--)
				{
					for (int neuronIdx = 0; neuronIdx < values[layerIdx].Count(); neuronIdx++)
					{
						double error = 0;
						for (int nextNeuronIdx = 0; nextNeuronIdx < values[layerIdx + 1].Count(); nextNeuronIdx++)
						{
							error += errors[layerIdx + 1][nextNeuronIdx] * weights[layerIdx][neuronIdx][nextNeuronIdx];
						}
						errors[layerIdx][neuronIdx] = error * (values[layerIdx][neuronIdx] > 0 ? 1 : 0);
					}
				}

				for (int layerIdx = values.Count() - 1; layerIdx > 0; layerIdx--)
				{
					for (int neuronIdx = 0; neuronIdx < values[layerIdx].Count(); neuronIdx++)
					{
						biases[layerIdx][neuronIdx] += errors[layerIdx][neuronIdx] * learningRate;
						for (int prevNeuronIdx = 0; prevNeuronIdx < values[layerIdx - 1].Count(); prevNeuronIdx++)
						{
							weights[layerIdx - 1][prevNeuronIdx][neuronIdx] += values[layerIdx - 1][prevNeuronIdx] * errors[layerIdx][neuronIdx] * learningRate;
						}
					}
				}
			}

			public void backPropagateSigmoid(double[] correctOutput)
			{
				for (int neuronIdx = 0; neuronIdx < values[values.Count() - 1].Count(); neuronIdx++)
				{
					errors[errors.Count() - 1][neuronIdx] = (correctOutput[neuronIdx] - values[values.Count() - 1][neuronIdx]) * values[values.Count() - 1][neuronIdx] * (1 - values[values.Count() - 1][neuronIdx]);
				}

				for (int layerIdx = values.Count() - 2; layerIdx > 0; layerIdx--)
				{
					for (int neuronIdx = 0; neuronIdx < values[layerIdx].Count(); neuronIdx++)
					{
						double error = 0;
						for (int nextNeuronIdx = 0; nextNeuronIdx < values[layerIdx + 1].Count(); nextNeuronIdx++)
						{
							error += errors[layerIdx + 1][nextNeuronIdx] * weights[layerIdx][neuronIdx][nextNeuronIdx];
						}
						errors[layerIdx][neuronIdx] = error * values[layerIdx][neuronIdx] * (1 - values[layerIdx][neuronIdx]);
					}
				}

				for (int layerIdx = values.Count() - 1; layerIdx > 0; layerIdx--)
				{
					for (int neuronIdx = 0; neuronIdx < values[layerIdx].Count(); neuronIdx++)
					{
						biases[layerIdx][neuronIdx] += errors[layerIdx][neuronIdx] * learningRate;
						for (int prevNeuronIdx = 0; prevNeuronIdx < values[layerIdx - 1].Count(); prevNeuronIdx++)
						{
							weights[layerIdx - 1][prevNeuronIdx][neuronIdx] += values[layerIdx - 1][prevNeuronIdx] * errors[layerIdx][neuronIdx] * learningRate;
						}
					}
				}
			}

			public void TrainTanh(double[] inputs, double[] correctOutput)
			{
				getTanhOutput(inputs);
				backPropagateTanh(correctOutput);
			}

			public void TrainReLU(double[] inputs, double[] correctOutput)
			{
				getReLUOutput(inputs);
				backPropagateReLU(correctOutput);
			}

			public void TrainSigmoid(double[] inputs, double[] correctOutput)
			{
				getSigmoidOutput(inputs);
				backPropagateSigmoid(correctOutput);
			}
		}
