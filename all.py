Q-1 Create two random integer matrices of dimension 5*6. Let them be X, Y. Display both matrices. Evaluate the following expression 12*X.XT + 15*(Y.YT)^2 + 108. Display the trace of the resulting matrix. Use numpy.

Q-2  Take Titanic kaggle CSV dataset. Find the number of children of age below 10 survived along with the total number of children of age below 10. Find the survival percentage of passengers per their passenger class they traveled. What is the percentage of survival between men and women! Use pandas.

Q-3 Solve the following nonlinear equations
x^2 + y^3 + 2xy+ 5 = 0
x+y-6 = 0
Use scipy

-------------ANswer------------

1).
import numpy as np


X = np.random.randint(1, 10, size=(5, 6))
Y = np.random.randint(1, 10, size=(5, 6))

print("Matrix X:")
print(X)
print("\nMatrix Y:")
print(Y)

res = 12*np.dot(X, X.T) + 15*np.dot(Y, Y.T)**2 + 108

print("\nResulting matrix:")
print(res)

trace = np.trace(res)
print("\nTrace of the resulting matrix:", trace)

2).
import pandas as pd

titanic_data = pd.read_csv("/content/titanic.csv")

children_survived = titanic_data[(titanic_data['Age'] < 10) & (titanic_data['Survived'] == 1)]
total_children_below_10 = titanic_data[titanic_data['Age'] < 10]

print("Number of children below 10 who survived: ",len(children_survived))
print("Total number of children below 10: ",len(total_children_below_10))

survival_percentage_by_class = titanic_data.groupby('Pclass')['Survived'].mean() * 100

print("\nSurvival percentage per passenger class:")
print(survival_percentage_by_class)

survival_percentage_by_sex = titanic_data.groupby('Sex')['Survived'].mean() * 100
print("\nSurvival percentage between men and women:")
print(survival_percentage_by_sex)

3).
from operator import eq
from scipy.optimize import fsolve

def equations(variables):
    x, y = variables
    eq1 = x**2 + y**3 + 2*x*y + 5
    eq2 = x + y - 6
    return [eq1, eq2]

initial_guess = [9, -5]
solution = fsolve(equations, initial_guess)
x_solution, y_solution = solution
print("Solution:")
print("x = ",{x_solution})
print("y = ",{y_solution})
print(equations(solution))

-----------------------------------------------------------------------------------------------------------
For opening image use pillow or opencv
Reference images are shared in the shared folder, work on the reference images only.
Q-1 From an image create a vector by flattening it completely. Similarly do for a few images and convert into a list of vectors as a Matrix. The final output should be a numpy Matrix.
Q-2 Compare two images using their histograms. Create histogram for two images in all three channels red, blue, green. Compare the images using histograms across each channel and find if they are equal. Don't use library functions.
Q-3 Take a reference image and apply a sobel filter on the image. Detect the edges and display the output. Don't use library functions for filters.
Q-4 Write algorithm from scratch without library for histogram of oriented gradients. Apply your algorithm for a reference image and show the resulting feature descriptors.

-------------ANswer------------

1).
import numpy as np
from PIL import Image

vector=[]

paths=['cat.100.jpg','cat.101.jpg','cat.102.jpg','cat.103.jpg','cat.104.jpg']

for path in paths:
  img=Image.open(path)
  flatten_image=np.array(img).flatten()
  flatten_image.resize(2200)
  vector.append(flatten_image)

v=np.vstack(vector)
print(v)

2).
import numpy as np
from PIL import Image

def hist(img):
  histogram=np.zeros((256,3),dtype=int)
  for row in img:
    for pix in row:
      r, g ,b=pix
      histogram[r][0]+=1
      histogram[g][1]+=1
      histogram[b][2]+=1
  return histogram

def compare(hist1,hist2):
  s=np.zeros(3,dtype=float)

  for i in range(256):
    s[0]+=min(hist1[i][0],hist2[i][0])
    s[1]+=min(hist1[i][1],hist2[i][1])
    s[2]+=min(hist1[i][2],hist2[i][2])

  total=np.sum(hist1)
  s/=total
  return s

img=Image.open('cat.100.jpg')
img2=Image.open('cat.101.jpg')

h1=hist(np.array(img))
h2=hist(np.array(img2))

print(compare(h1,h2))

3).
from PIL import Image
import numpy as np
from scipy.signal import convolve2d

input_image_path = 'cat.100.jpg'
output_image_path = 'output_image.jpg'
image = Image.open(input_image_path)

gray_image = image.convert("L")

image_array = np.array(gray_image)

sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

filtered_x = convolve2d(image_array, sobel_x, mode='same', boundary='wrap')
filtered_y = convolve2d(image_array, sobel_y, mode='same', boundary='wrap')

edge_image = np.sqrt(filtered_x**2 + filtered_y**2)

edge_image = (edge_image / np.max(edge_image)) * 255

edge_image_pil = Image.fromarray(edge_image.astype(np.uint8))

edge_image_pil.save(output_image_path)

print("Edge-detected image saved!")

4).
import numpy as np
import cv2
import matplotlib.pyplot as plt


img = cv2.imread("cat.100.jpg", cv2.IMREAD_GRAYSCALE)
cell = 8
block =2

#gradiemts
sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
gradx = cv2.filter2D(img, cv2.CV_64F, sobelx)
grady = cv2.filter2D(img, cv2.CV_64F, sobely)

#magnitude
mag = np.sqrt(gradx**2 + grady**2)

#dir
angle = np.arctan2(grady, gradx) * (180 / np.pi)
angle[angle < 0] += 180

print(mag.shape, angle.shape)

cell_x = mag.shape[0]
cell_y = mag.shape[1]

def create_hist(m,a):
    hist = np.zeros(9)
    bin_width = 20

    for i in range(m.shape[0]):
        for j in range(a.shape[1]):
            new_a = a[i, j]
            weight = (new_a % bin_width) / bin_width
            bin_idx = int(new_a // bin_width) % 9
            hist[bin_idx] += (1 - weight) * m[i, j]
            hist[(bin_idx + 1) % 9] += weight * m[i, j]
    return hist

hists = np.zeros((cell_x,cell_y,9))

for i in range(cell_x):
  for j in range(cell_y):
    cell_mag = mag[i*cell:(i+1)*cell, j*cell:(j+1)*cell]
    cell_angle = angle[i*cell:(i+1)*cell, j*cell:(j+1)*cell]
    hists[i, j, :] = create_hist(cell_mag, cell_angle)
#print(hists)


hog_features = []

for i in range(cell_x - block + 1):
    for j in range(cell_y - block + 1):
        block_histograms = hists[i:i+block, j:j+block, :]
        normalized_block = block_histograms / np.sqrt(np.sum(block_histograms ** 2) + 1e-6)
        hog_features.append(normalized_block.flatten())

#print(np.array(hog_features))

hog = np.array(hog_features)

plt.tight_layout()
fig = plt.figure(figsize=(8,8))
fig.add_subplot(1,2,1)
plt.imshow(img, cmap = "gray")
plt.axis('off')
plt.title("original image")
orient = angle*np.pi/180.0
orient[mag<50]=np.nan
fig.add_subplot(1,2,2)
plt.imshow(orient, cmap = "hsv")
plt.axis('off')
plt.show()

orient.shape

-----------------------------------------------------------------------------------------------------
Q-1 Create Tic Tac Toe Game.

-------------ANswer------------

import tensorflow as tf
import numpy as np
import random'

class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.players = ['X', 'O']
        self.current_player = None
        self.winner = None
        self.game_over = False

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = None
        self.winner = None
        self.game_over = False

    def available_moves(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    moves.append((i, j))
        return moves

    def make_move(self, move):
        if self.board[move[0]][move[1]] != 0:
            return False
        self.board[move[0]][move[1]] = self.players.index(self.current_player) + 1
        self.check_winner()
        self.switch_player()
        return True

    def switch_player(self):
        if self.current_player == self.players[0]:
            self.current_player = self.players[1]
        else:
            self.current_player = self.players[0]

    def check_winner(self):
        # Check rows
        for i in range(3):
            if self.board[i][0] == self.board[i][1] == self.board[i][2] != 0:
                self.winner = self.players[int(self.board[i][0] - 1)]
                self.game_over = True
        # Check columns
        for j in range(3):
            if self.board[0][j] == self.board[1][j] == self.board[2][j] != 0:
                self.winner = self.players[int(self.board[0][j] - 1)]
                self.game_over = True
        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            self.winner = self.players[int(self.board[0][0] - 1)]
            self.game_over = True
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            self.winner = self.players[int(self.board[0][2] - 1)]
            self.game_over = True

    def print_board(self):
        print("-------------")
        for i in range(3):
            print("|", end=' ')
            for j in range(3):
                print(self.players[int(self.board[i][j] - 1)] if self.board[i][j] != 0 else " ", end=' | ')
            print()
            print("-------------")

game = TicTacToe()
game.current_player = game.players[0]
game.print_board()

while not game.game_over:
    move = input(f"{game.current_player}'s turn. Enter row and column (e.g. 0 0): ")
    move = tuple(map(int, move.split()))
    while move not in game.available_moves():
        move = input("Invalid move. Try again: ")
        move = tuple(map(int, move.split()))
    game.make_move(move)
    game.print_board()

if game.winner:
    print(f"{game.winner} wins!")
else:
    print("It's a tie!")


-----------------------------------------------------------------------------------------------------
Q-1 MP neuron for And, Or, Not gates
Q-2 Perceptron for And, Or and Xor
Q-3 Tic Tac Toe toe game similar to checkers (Same Lab-4 Task)

-------------ANswer------------

1).
import numpy as np
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
w1 = [1, 1, 1, 1]
w2 = [1, 1, 1, 1]
#and

t=2
print("x1    x2    O")
for i in range(len(x1)):
    if ( x1[i]*w1[i] + x2[i]*w2[i] ) >= t:
        print(x1[i],'   ',x2[i],'   ', 1)
    else:
        print(x1[i],'   ',x2[i],'   ', 0)

#or
t=1
print("x1    x2    O")
for i in range(len(x1)):
    if ( x1[i]*w1[i] + x2[i]*w2[i] ) >= t:
        print(x1[i],'   ',x2[i],'   ', 1)
    else:
        print(x1[i],'   ',x2[i],'   ', 0)
#not
x = [0, 1]
w = [-1, -1]
t = 0

print("x      w     t     O")
for i in range(len(x)):
    if ( x[i]*w[i]) >= t:
        print(x[i],'   ',w[i],'   ',t,'   ', 1)
    else:
        print(x[i],'   ',w[i],'   ',t,'   ', 0)



2).
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
#and
t=2
print("x1    x2    O")
for i in range(len(x1)):
    if ( x1[i] + x2[i] ) >= t:
        print(x1[i],'   ',x2[i],'   ', 1)
    else:
        print(x1[i],'   ',x2[i],'   ', 0)
#or
t=1
print("x1    x2    O")
for i in range(len(x1)):
    if ( x1[i] + x2[i] ) >= t:
        print(x1[i],'   ',x2[i],'   ', 1)
    else:
        print(x1[i],'   ',x2[i],'   ', 0)
#xor
t = 1

print("x1    x2    O")
for i in range(len(x1)):
    if x1[i] + x2[i] == t:
        print(x1[i],'   ',x2[i],'   ', 1)
    else:
        print(x1[i],'   ',x2[i],'   ', 0)
#not
x = [0, 1]
t = 1

print("x      O")
for i in range(len(x)):
    if ( x[i]) < t:
        print(x[i],'   ', 1)
    else:
        print(x[i],'   ', 0)
#AND

def step_function(x):
    return 1 if x >= 0 else 0

and_gate_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
and_gate_outputs = np.array([0, 0, 0, 1])

weights = np.random.rand(2)
bias = np.random.rand()

learning_rate = 0.1
epochs = 1000

for epoch in range(epochs):
    for i in range(len(and_gate_inputs)):
        input_vector = and_gate_inputs[i]
        target_output = and_gate_outputs[i]

        net_input = np.dot(input_vector, weights) + bias
        predicted_output = step_function(net_input)

        error = target_output - predicted_output

        weights += learning_rate * error * input_vector
        bias += learning_rate * error

for i in range(len(and_gate_inputs)):
    input_vector = and_gate_inputs[i]
    output = step_function(np.dot(input_vector,weights)+bias)
    print(f"Input: {input_vector}, Weight: {weights} Output: {output}")



#OR

def step_function(x):
    return 1 if x >= 0 else 0

and_gate_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
and_gate_outputs = np.array([0, 1, 1, 1])

weights = np.random.rand(2)
bias = np.random.rand()

learning_rate = 0.1
epochs = 1000

for epoch in range(epochs):
    for i in range(len(and_gate_inputs)):
        input_vector = and_gate_inputs[i]
        target_output = and_gate_outputs[i]

        net_input = np.dot(input_vector, weights) + bias
        predicted_output = step_function(net_input)

        error = target_output - predicted_output

        weights += learning_rate * error * input_vector
        bias += learning_rate * error

for i in range(len(and_gate_inputs)):
    input_vector = and_gate_inputs[i]
    output = step_function(np.dot(input_vector,weights)+bias)
    print(f"Input: {input_vector}, Weight: {weights} Output: {output}")



#NOT

def step_function(x):
    return 1 if x >= 0 else 0

and_gate_inputs = np.array([[0], [1]])
and_gate_outputs = np.array([1, 0])

weights = np.random.rand(1)
bias = np.random.rand()

learning_rate = 0.1
epochs = 1000

for epoch in range(epochs):
    for i in range(len(and_gate_inputs)):
        input_vector = and_gate_inputs[i]
        target_output = and_gate_outputs[i]

        net_input = np.dot(input_vector, weights) + bias
        predicted_output = step_function(net_input)

        error = target_output - predicted_output

        weights += learning_rate * error * input_vector
        bias += learning_rate * error

for i in range(len(and_gate_inputs)):
    input_vector = and_gate_inputs[i]
    output = step_function(np.dot(input_vector,weights)+bias)
    print(f"Input: {input_vector}, Weight: {weights} Output: {output}")

---------------------------------------------------------------------------------------------------------------
Q-1 Back propagation of one hidden layer neural network without using libraries. Fit the neural network for XOR gate data.


-------------ANswer------------

1).
import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([[0], [1], [1], [0]])
input_size = 2
hidden_size = 2
output_size = 1
learning_rate = 0.1
iteration = 10000
np.random.seed(0)
input_layer_weights = np.random.uniform(size=(input_size, hidden_size))
hidden_layer_weights = np.random.uniform(size=(hidden_size, output_size))
input_layer_bias = np.zeros((1, hidden_size))
hidden_layer_bias = np.zeros((1, output_size))
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)
for i in range(iteration):

    #forward propogation
    hidden_layer_input = np.dot(X, input_layer_weights) + input_layer_bias
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, hidden_layer_weights) + hidden_layer_bias
    output_layer_output = sigmoid(output_layer_input)

    error = Y - output_layer_output

    #error calulation
    d_output = error * sigmoid_derivative(output_layer_output)
    error_hidden_layer = d_output.dot(hidden_layer_weights.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)


    #weight and bias updation
    hidden_layer_weights += hidden_layer_output.T.dot(d_output) * learning_rate
    hidden_layer_bias += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    input_layer_weights += X.T.dot(d_hidden_layer) * learning_rate
    input_layer_bias += np.sum(d_hidden_layer, axis=0, keepdims=True) * learning_rate
test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicted_output = sigmoid(np.dot(sigmoid(np.dot(test_input, input_layer_weights) + input_layer_bias), hidden_layer_weights) + hidden_layer_bias)

filter_arr= [1 if x>0.5 else 0 for x in predicted_output]
print(filter_arr)

print(predicted_output)


_______________________________________________________________________________________________________________

Q-1 Use k-means clustering to segment a color image without using any library. The value of k should be varied and the resulting segmented image output should be shown in color. Each color for each cluster. Show the difference in the segmented image from a lower k to a higher k.


-------------ANswer------------

1).
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
img = Image.open("/content/butterfly.jpeg")
img = np.array(img)
def kmeans(img, k, maxiter=  100):
  pixels = np.array(img).reshape(-1,3)
  cent = pixels[np.random.choice(pixels.shape[0],k,replace=False)]
  for i in range(maxiter):
    dist = np.linalg.norm(pixels[:, np.newaxis]-cent, axis=2)
    label = np.argmin(dist,axis=1)
    newcent = np.array([pixels[label==i].mean(axis=0) for i in range(k)])
    if np.all(cent == newcent):
      break
    cent = newcent
  seg = cent[label].reshape(img.shape)
  return seg
l = []
for k in range(3,11):
  segimg = kmeans(img,k)
  l.append(segimg)
plt.imshow(img)
fig = plt.figure(figsize=(20,10))
for i in range(len(l)):
  fig.add_subplot(2,4,i+1)
  plt.imshow((l[i] * 255).astype(np.uint8))
  plt.axis('off')
  plt.title(f"K = {i+3}")
plt.show()
seg = kmeans(img,4)
plt.imshow((seg * 255).astype(np.uint8))
plt.show()

--------------------------------------------------------------------------------------------------------------
Q-1 Do PCA on a grayscale image of higher resolution. Do PCA along the x direction and the y direction. Reconstruct both the images from their compressed images along the x axis and y axis. Do not use the PCA library, use numpy.


-------------ANswer------------
1).

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
img = Image.open("/content/butterfly.jpeg").convert('L')
arr = np.array(img)
meanx = np.mean(arr,axis=1)
centx = arr - meanx[:,np.newaxis]
ux,sx,vhx = np.linalg.svd(centx,full_matrices = False)
meany = np.mean(arr,axis=0)
centy = arr - meany
uy,sy,vhy = np.linalg.svd(centy,full_matrices = False)
vhx.shape
num_components_list = [5, 10, 20, 50]

plt.figure(figsize=(15, 10))

for i, num_components in enumerate(num_components_list, start=1):
    compressed_x = ux[:, :num_components] @ np.diag(sx[:num_components]) @ vhx[:num_components, :]
    reconstructed_x = meanx[:, np.newaxis] + compressed_x
    compressed_y = uy[:, :num_components] @ np.diag(sy[:num_components]) @ vhy[:num_components, :]
    reconstructed_y = meany + compressed_y
    plt.subplot(4, 3, (i-1)*3 + 1)
    plt.imshow(reconstructed_x, cmap='gray')
    plt.title(f'X-axis, {num_components} components')
    plt.axis('off')

    plt.subplot(4, 3, (i-1)*3 + 2)
    plt.imshow(reconstructed_y, cmap='gray')
    plt.title(f'Y-axis, {num_components} components')
    plt.axis('off')


plt.tight_layout()
plt.show()
arr.shape
plt.imshow(arr[0].reshape(64,64), cmap='gray')
plt.axis('off')
--------------------------------------------------------------------------------------------------------------

Q-1 Implement SVM.
Q-2 Implement Decision Tree.

-------------ANswer------------

1).

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
mnist = datasets.fetch_openml('mnist_784', version=1)
X = mnist.data
y = mnist.target
X.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
svm_classifier = SVC(C=1.0, kernel='linear', random_state=101)
svm_classifier.fit(X_train, y_train)
svm_classifier
y_pred = svm_classifier.predict(X_test)
y_pred
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
-------------------------------------------------------------------------------------------------------------

Q-1 Expectations Maximization


-------------ANswer------------



------------PCA------
import numpy as np

def pca(X, num_components):
    # Standardize the data
    X_std = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Compute the covariance matrix
    cov_matrix = np.cov(X_std, rowvar=False)

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Select the top 'num_components' eigenvectors
    top_eigenvectors = eigenvectors[:, :num_components]

    # Project the original data onto the new feature space
    X_pca = np.dot(X_std, top_eigenvectors)

    return X_pca

# Example usage
# Generate random data for demonstration
np.random.seed(42)
data = np.random.rand(100, 3)

# Perform PCA with 2 components
result = pca(data, num_components=2)

print("Original data shape:", data.shape)
print("Transformed data shape:", result.shape)



----------EMA-----------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate a dataset with two Gaussian components
mu1, sigma1 = 2, 1
mu2, sigma2 = -1, 0.8
X1 = np.random.normal(mu1, sigma1, size=200)
X2 = np.random.normal(mu2, sigma2, size=600)
X = np.concatenate([X1, X2])

# Plot the density estimation using seaborn
sns.kdeplot(X)
plt.xlabel('X')
plt.ylabel('Density')
plt.title('Density Estimation of X')
plt.show()


# Initialize parameters
mu1_hat, sigma1_hat = np.mean(X1), np.std(X1)
mu2_hat, sigma2_hat = np.mean(X2), np.std(X2)
pi1_hat, pi2_hat = len(X1) / len(X), len(X2) / len(X)


# Perform EM algorithm for 20 epochs
num_epochs = 20
log_likelihoods = []

for epoch in range(num_epochs):
	# E-step: Compute responsibilities
	gamma1 = pi1_hat * norm.pdf(X, mu1_hat, sigma1_hat)
	gamma2 = pi2_hat * norm.pdf(X, mu2_hat, sigma2_hat)
	total = gamma1 + gamma2
	gamma1 /= total
	gamma2 /= total
	
	# M-step: Update parameters
	mu1_hat = np.sum(gamma1 * X) / np.sum(gamma1)
	mu2_hat = np.sum(gamma2 * X) / np.sum(gamma2)
	sigma1_hat = np.sqrt(np.sum(gamma1 * (X - mu1_hat)**2) / np.sum(gamma1))
	sigma2_hat = np.sqrt(np.sum(gamma2 * (X - mu2_hat)**2) / np.sum(gamma2))
	pi1_hat = np.mean(gamma1)
	pi2_hat = np.mean(gamma2)
	
	# Compute log-likelihood
	log_likelihood = np.sum(np.log(pi1_hat * norm.pdf(X, mu1_hat, sigma1_hat)
								+ pi2_hat * norm.pdf(X, mu2_hat, sigma2_hat)))
	log_likelihoods.append(log_likelihood)

# Plot log-likelihood values over epochs
plt.plot(range(1, num_epochs+1), log_likelihoods)
plt.xlabel('Epoch')
plt.ylabel('Log-Likelihood')
plt.title('Log-Likelihood vs. Epoch')
plt.show()

# Plot the final estimated density
X_sorted = np.sort(X)
density_estimation = pi1_hat*norm.pdf(X_sorted,
										mu1_hat, 
										sigma1_hat) + pi2_hat * norm.pdf(X_sorted,
																		mu2_hat, 
																		sigma2_hat)


plt.plot(X_sorted, gaussian_kde(X_sorted)(X_sorted), color='green', linewidth=2)
plt.plot(X_sorted, density_estimation, color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('Density')
plt.title('Density Estimation of X')
plt.legend(['Kernel Density Estimation','Mixture Density'])
plt.show()










