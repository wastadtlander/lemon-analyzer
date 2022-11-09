using FileIO, Images, MosaicViews, Plots, Random

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
    println("Splitting Data")

    n = size(D, 1)
    idx = shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, percent * n))
    test_idx = view(idx, (floor(Int, percent * n) + 1):n)
    return D[train_idx,:], D[test_idx,:], T[train_idx,:], T[test_idx,:]
end

function main()
    # Default Directory
    cd()
    cd("./Documents/ML Project/")

    # Paths of Data
    goodQualityPath = "./lemon_dataset/good_quality/"
    badQualityPath = "./lemon_dataset/bad_quality"
    emptyBackgroundPath = "./lemon_dataset/empty_background"

    # Load Data as Vector{Matrix{RGB{N0f8}}}
    goodQuality, badQuality, emptyBackground = load_images(goodQualityPath, badQualityPath, emptyBackgroundPath)

    # Display Samples of Data
    mosaicview([goodQuality[1:3]; badQuality[1:3]; emptyBackground[1:3]], fillvalue=0.5, npad=2, nrow=3, rowmajor=true)

    # Cleaning Data
    D, T = clean_data(goodQuality, badQuality, emptyBackground)

    # Splitting Data
    DTrain, DTest, TTrain, TTest = split_data(D, T, 0.66)

    println(size(DTrain))
    println(size(DTest))
    println(size(TTrain))
    println(size(TTest))

    mosaicview(DTrain[1:81], fillvalue=0.5, npad=2, nrow=9, rowmajor=true)
end

main()