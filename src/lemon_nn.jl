using BenchmarkTools, FileIO, Images, MosaicViews, Plots, Random

function load_images(goodQualityPath, badQualityPath, emptyBackgroundPath)
    # Load Good Quality
    println("Loading Good Quality Images")

    cd(goodQualityPath)
    goodQualityPaths = readdir()
    goodQualityPathsSize = size(goodQualityPaths, 1)
    goodQualityImgs = Vector{Matrix{RGB{N0f8}}}(undef, goodQualityPathsSize)
    for (index, goodQualityImg) in enumerate(goodQualityPaths)
        img = load(goodQualityImg)
        goodQualityImgs[index] = img
    end

    # Load Bad Quality
    println("Loading Bad Quality Images")

    cd("../..")
    cd(badQualityPath)
    badQualityPaths = readdir()
    badQualityPathsSize = size(badQualityPaths, 1)
    badQualityImgs = Vector{Matrix{RGB{N0f8}}}(undef, badQualityPathsSize)
    for (index, badQualityImg) in enumerate(badQualityPaths)
        img = load(badQualityImg)
        badQualityImgs[index] = img
    end

    # Load Empty Background
    println("Loading Empty Background Images")

    cd("../..")
    cd(emptyBackgroundPath)
    emptyBackgroundPaths = readdir()
    emptyBackgroundPathsSize = size(emptyBackgroundPaths, 1)
    emptyBackgroundImgs = Vector{Matrix{RGB{N0f8}}}(undef, emptyBackgroundPathsSize)
    for (index, emptyBackgroundImg) in enumerate(emptyBackgroundPaths)
        img = load(emptyBackgroundImg)
        emptyBackgroundImgs[index] = img
    end

    return goodQualityImgs, badQualityImgs, emptyBackgroundImgs
end

function clean_data(goodQuality, badQuality, emptyBackground)
    println("Cleaning Data")

    # Concatenates Data
    D = [goodQuality; badQuality; emptyBackground]

    DSize = size(D, 1)

    # Converts RGB to Grayscale 
    for i = 1:DSize
        D[i] = Gray.(D[i])
    end

    # Builds Targets Corresponding to Data
    # 1: Good Quality
    # 2: Bad Quality
    # 3: Empty Background
    goodQualitySize = size(goodQuality, 1)
    badQualitySize = size(badQuality, 1)
    emptyBackgroundSize = size(emptyBackground, 1)
    T = zeros(goodQualitySize + badQualitySize + emptyBackgroundSize)

    TSize = size(T, 1)

    for i = 1:TSize
        if i <= goodQualitySize
            T[i] = 1
        elseif i <= goodQualitySize + badQualitySize
            T[i] = 2
        else
            T[i] = 3
        end
    end 

    return D, T
end

function split_data(D, T, percent)
    n = size(D, 1)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, percent * n))
    test_idx = view(idx, (floor(Int, percent * n) + 1):n)
    return D[train_idx,:], D[test_idx,:], T[train_idx,:], T[test_idx,:]
end

function train(Data, Target, numClasses)
    # Number of Training Samples
    N = length(Target)

    # Dimesion of Input
    D = length(Data[1])

    # Setting Up Training and Validation Set
    # println("Splitting Train -> Train & Validation")
    println("Splitting -> Train & Validation")
    valNum = round(Int64, N / 3)
    trainNum = N - valNum

    DTrain, DVal, TTrain, TVal = split_data(Data, Target, trainNum / N)

    # Number of Hidden Nodes
    hiddenNodes = 2

    # Batch Size
    batchSize = floor(Int, trainNum / 10)

    # Initial Adam (Hyperparameters) Parameters
    mOne = zeros(hiddenNodes, D)
    mTwo = zeros(numClasses, hiddenNodes + 1) # To account for bias node

    vOne = zeros(hiddenNodes, D)
    vTwo = zeros(numClasses, hiddenNodes + 1) # To account for bias node

    alpha = 1 / 1000 # Step
    betaOne = 0.9 # Decay rate 1
    betaTwo = 0.999 # Decay rate 2
    epsilon = 1e-8 # Numerical stability param

    # Setting Up Network Layers
    inputLayer = zeros(D)
    hiddenLayer = zeros(hiddenNodes)
    outputLayer = zeros(3)

    layerOneWeight = 5 * randn(hiddenNodes, D)
    layerTwoWeight = 5 * randn(numClasses, hiddenNodes + 1) # To account for bias nodes

    # Pre-network information
    println("Hidden Nodes: ", hiddenNodes)
    println("Batch Size: ", batchSize)
    println("Total Weights: ", prod(size(layerOneWeight)) + prod(size(layerTwoWeight)))
    println("Initial Training Error: ", test(DTrain, TTrain, layerOneWeight, layerTwoWeight, 3))
    println("Initial Validation Error: ", test(DVal, TVal, layerOneWeight, layerTwoWeight, 3))

    # Network Information
    epoch = 1
    stop = false
    while !stop
        # Copys of Weights to Calculate Norm Update
        layerOnePreviousWeight = copy(layerOneWeight)
        layerTwoPreviousWeight = copy(layerTwoWeight)

        iteration = 1
        gradientOne = zeros(hiddenNodes, D)
        gradientTwo = zeros(numClasses, hiddenNodes + 1) # To account for bias node

        trainIndex = shuffle(1:trainNum)

        offset = -1
        for b = 1:(floor(Int, trainNum / batchSize) - 1)
            offset += batchSize
            for n = 1:batchSize
                x = DTrain[trainIndex[offset + n]]
                t = TTrain[trainIndex[offset + n]]

                inputLayer = x
                # Forward Propogation
                y = zeros(hiddenNodes + 1) # To account for bias nodes
                y[1] = 1 # Bias Node
                for j = 1:hiddenNodes
                    hiddenLayer[j] = 0
                    for k = 1:D
                        hiddenLayer[j] += layerOneWeight[j, k] * inputLayer[k].r
                    end
                    y[j + 1] = sigmoidActivation(hiddenLayer[j])
                end

                outputLayer = zeros(3)
                for c = 1:numClasses
                    for j = 1:hiddenNodes + 1 # To account for bias nodes
                        outputLayer[c] += layerTwoWeight[c, j] * y[j]
                    end
                end

                z = zeros(3)
                for c = 1:numClasses
                    z[c] = linearActivation(outputLayer[c])
                end

                modifiedT = zeros(3)
                if t == 1
                    modifiedT[1] = 1
                elseif t == 2
                    modifiedT[2] = 1
                else
                    modifiedT[3] = 1
                end

                delta = z - modifiedT

                # Gradients
                for c = 1:numClasses
                    for j = 1:hiddenNodes + 1
                        if j == 1
                            gradientTwo[c, j] += delta[c] * y[j]
                        else
                            gradientTwo[c, j] += delta[c] * y[j] * linearActivationDerivative(outputLayer[c])
                        end
                    end
                end

                for c = 1:numClasses
                    for i = 1:D
                        for j = 1:hiddenNodes
                            gradientOne[j, i] += delta[c] * layerTwoWeight[c, j + 1] * sigmoidActivationDerivative(hiddenLayer[j]) * inputLayer[i].r
                        end
                    end
                end

                # ADAM
                mOne = betaOne .* mOne + (1 - betaOne) .* gradientOne
                mTwo = betaOne .* mTwo + (1 - betaOne) .* gradientTwo

                vOne = betaTwo .* vOne + (1 - betaTwo) .* (gradientOne .^ 2)
                vTwo = betaTwo .* vTwo + (1 - betaTwo) .* (gradientTwo .^ 2)

                mOneHat = mOne ./ (1 - betaOne ^ iteration)
                mTwoHat = mTwo ./ (1 - betaOne ^ iteration)

                vOneHat = vOne ./ (1 - (betaTwo ^ iteration))
                vTwoHat = vTwo ./ (1 - (betaTwo ^ iteration))

                iteration += 1

                # Weights
                for c = 1:numClasses
                    for j = 1:hiddenNodes + 1
                        layerTwoWeight[c, j] -= alpha * mTwoHat[c, j] / (sqrt(vTwoHat[c, j]) + epsilon)
                    end
                end

                for i = 1:D
                    for j = 1:hiddenNodes
                        layerOneWeight[j, i] -= alpha * mOneHat[j, i] / (sqrt(vOneHat[j, i]) + epsilon)
                    end
                end
            end
        end

        norm = 0
        for c = 1:numClasses
            for j = 1:hiddenNodes + 1 # To account for bias nodes
                dw = layerTwoPreviousWeight[c, j] - layerTwoWeight[c, j]
                norm += dw^2
            end
        end
        for i = 1:D
            for j = 1:hiddenNodes
                dw = layerOnePreviousWeight[j, i] - layerOneWeight[j, i]
                norm += dw^2
            end
        end

        println("Epoch ", epoch, " Training Error Is ", test(DTrain, TTrain, layerOneWeight, layerTwoWeight, 3))
        println("Norm is ", norm)

        epoch += 1

        stop = norm < 1e-6
    end

    # Post-network Information
    println("Final Training Error: ", test(DTrain, TTrain, layerOneWeight, layerTwoWeight, 3))
    println("Final Validation Error: ", test(DVal, TVal, layerOneWeight, layerTwoWeight, 3))
end

function test(Data, Target, layerOneWeight, layerTwoWeight, numClasses)
    # Number of Training Samples
    N = length(Target)

    # Dimesion of Input
    D = length(Data[1])

    # Number of Hidden Nodes
    hiddenNodes = size(layerOneWeight)[1]

    # Calculate Error
    error = 0
    for n = 1:N
        x = Data[n]
        t = Target[n]

        # Forward Propogation
        y = zeros(hiddenNodes + 1) # To account for bias nodes
        y[1] = 1 # Bias Node
        for j = 1:hiddenNodes
            a = 0
            for k = 1:D
                a += layerOneWeight[j, k] * x[k].r
            end
            y[j + 1] = sigmoidActivation(a)
        end

        a = zeros(3)
        z = zeros(3)
        for c = 1:numClasses
            for j = 1:hiddenNodes + 1 # To account for bias nodes
                a[c] += layerTwoWeight[c, j] * y[j]
            end
            z[c] = softMax(a, a[c], numClasses)
            if t == c
                error += 1 * log(z[c])
            end
        end
    end

    return -error
end

function linearActivation(a)
    return a
end

function linearActivationDerivative(a)
    return 1
end

function sigmoidActivation(a)
    return 1 / (1 + exp(-a))
end

function sigmoidActivationDerivative(a)
    return sigmoidActivation(a) * (1 - sigmoidActivation(a))
end

function softMax(a, x, numClasses)
    aq = 0
    for c = 1:numClasses
        aq += exp(sigmoidActivationDerivative(a[c]))
    end
    return exp(sigmoidActivationDerivative(x)) / aq
end

function main()
    # Default Directory
    cd()
    cd("./Documents/ML Project/")

    # Paths of Data
    goodQualityPath = "./lemon_dataset_scaled/good_quality/"
    badQualityPath = "./lemon_dataset_scaled/bad_quality"
    emptyBackgroundPath = "./lemon_dataset_scaled/empty_background"

    # Load Data as Vector{Matrix{RGB{N0f8}}}
    goodQuality, badQuality, emptyBackground = load_images(goodQualityPath, badQualityPath, emptyBackgroundPath)

    # Display Samples of Data
    # mosaicview([goodQuality[1:3]; badQuality[1:3]; emptyBackground[1:3]], fillvalue=0.5, npad=2, nrow=3, rowmajor=true)

    # Cleaning Data
    D, T = clean_data(goodQuality, badQuality, emptyBackground)

    # Splitting Data
    # println("Splitting Data -> Train & Test")
    # DTrain, DTest, TTrain, TTest = split_data(D, T, 0.66)

    # Display Cleaned Samples of Data
    # mosaicview(DTrain[1:81], fillvalue=0.5, npad=2, nrow=9, rowmajor=true)

    # train(DTrain, TTrain)
    train(D, T, 3)
end

# @btime main()
main()