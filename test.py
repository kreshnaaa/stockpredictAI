''' you can explore more advanced features like handling forms, 
      database integration, user authentication, and more.'''

'''REST=Representational state transfer'''

import pandas as pd
import numpy as np

# Create a Pandas DataFrame
data = {
    'A': [1, 2, 3],
    'B': [4, 5, 6],
    'C': [7, 8, 9]
}

df = pd.DataFrame(data)

# Using .to_numpy() method
numpy_array_1 = df.to_numpy()

# Using .values attribute
numpy_array_2 = df.values

# Print the resulting NumPy arrays
print("Using .to_numpy():")
print(numpy_array_1)

print("\nUsing .values attribute:")
print(numpy_array_2)



