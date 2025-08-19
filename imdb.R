#sentiment analysis using RNN for movie reviews
library(keras)
library(reticulate)
conda_list()
use_condaenv("tf", required=T)
imdb = dataset_imdb(num_words = 500) # A review may have more than 500 words. But here I am using only the most frequent 500 words in the analysis. Will ignore words after those.

# Split the dataset into train and test
str(imdb)
trainX = imdb$train$x
trainY = imdb$train$y
testX = imdb$test$x
testY = imdb$test$y

# View a sample review
review2 = trainX[[2]]
review2 # consists of multiple numbers, we need to find what are those corresponding words, for which we need the original word index
wordIndex = dataset_imdb_word_index()
wordIndex #words and corresponding numbers - we can use words to get the numbers
# review2 has some numbers which we can convert into words.
# 1 indicates the beginning of a review, it is not corresponding to any word
# 0 is the padding - If the review is shorter than 500, rest will be padded with 0s
# 2 is for unknown words which are out of the 500 words we have chosen
review2 = review2[review2>=3] #greater than or equal 3 will give only the meaningful words inside the review
review2 = review2-3 #all numbers shown are already increased by 3, to get the index corresponding to vocab wordIndex which starts from 1, we have to do -3 from each word.

# sorting word index based on frequency
head(wordIndex)
indexedWords = names(sort(unlist(wordIndex)))
names(wordIndex[wordIndex %in% review2])

#function to generate original message word by word
toSentence = function(numbers){
  result = ""
  for(n in numbers){
    if(n < 3)
      result = cat(result, "?")
    else
    {
      result = cat(result, indexedWords[n-3]) #n-3 to remove unknowns and beginnings
    }
  }
  return(result)
}

#check some reviews
review3 = trainX[[3]]
toSentence(review3)

#padding the reviews
trainX = pad_sequences(trainX, 150)
testX = pad_sequences(testX, 150) #every msg should be atleast 150 words, if not pad with 0s

#building the model
model = keras_model_sequential()
layer_embedding( model, input_dim = 500, output_dim = 32)
# inputdim = total number of words in vocab = 500
# outoutdim = numbers in the vector for each word
layer_simple_rnn(model, units = 128)
layer_dense(model, units = 1, activation = "sigmoid") #output layer
summary(model)

#compile and train the model
compile(model, optimizer = "adam", loss="binary_crossentropy", metrics = "accuracy")
history = fit(model, trainX, trainY, epochs = 15, batch_size = 128)
