# House Rocket - Insight Project


## 1. Business Question


  House Rocket is a fictitious company based on Seatle , WA. It is a real state company with a core business of buy, renovated and sell houses or apartments, their CEO in a pursue to maximize their profit are looking for an analysis in their dataset to find the best businesses available to them, buying at a low price, renovate and sell it to a higher price, also want to know what is the best price they could sell to make most profit possible. He wants the answer for two questions:
  
     1. Which is the real state that House Rocket should buy and at what price?
     
     2. Once bought, when is the best moment to sell and at what price?
     
## 2. Business Assumptions

  - All the data assumptions are based on the dataset available
  - The data are from in Seattle, WA real state market
  
## 3. Solution Strategy

  I divided the solution strategy into the two questions:
  
     1. Which is the real state that House Rocket should buy and at what price?
     
      - Group the data by zipcode.
      - Find the median price by each zipcode.
      - The house price lower than the median and condition greater than 3, 
      is a strong suggestion to buy.
     
     2. Once bought, when is the best moment to sell and at what price?
      
      - Group the data by zipcode and season
      - Find the price median by zipcode and season
      - The buy price lower than the price median by season and zipcode adds a 30% increase, 
      that would be the sell price with a profit of 30%.
      - The buy price greater than the price median by season and zipcode adds 10% increase, 
      that would be the sell price with a profit of 10%.
 
## 4. Top 3 Data Insights

  1. Houses are 62% more expensive in the summer than in the winter.
  2. Houses with year built less than 1955 are very slightly cheaper than those with year built greater than 1955.
  3. The real state market had a decrease price about 50% in 2015 compared to 2014.
  
## 5. Business Results

  After analyzing all this dataset looking for the best businesses that could be made,
 let's take a look at how House Rocket could really make profit out of it.

|    Season      |     Profit      |
|  ------------- | -------------   |
|    Autumn      |   214,012,611   |  
|    Spring      |   244,234,470   |  
|    Summer      |   260,182,320   |  
|    Winter      |   145,300,366   |

## 6. Conclusion

  As we could by this data, the best season to sell a house/apartment is in the summer, on the other hand
 the best moment to buy a house/apartment is in the winter. This is a great insight for House Rocket and it is
 possible to make a very good profit by this.
 
## 7. Lessons Learned
## 8. Next Steps
  
  The next steps I would take for this project is, create more hypothesis, test new graphs and colors, and use a 
 machine learning model to find the best price to buy and sell.
 
 
 # Links:
  
  datasource: https://www.kaggle.com/shivachandel/kc-house-data
  
  webapp: https://insight-project-v1.herokuapp.com/
