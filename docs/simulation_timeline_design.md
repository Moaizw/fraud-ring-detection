## Simulation Calendar

- ONS data from 2025 therefore it's appropriate for my simulation model to use 2025 as the time window. 
- Proposed 9-month window window: 6th Jan 2025 (Monday) -> 26th Sep 2025 (Friday), last week of the month (payday).
- Train/Test split: 6 month training window (6th Jan -> 27th Jun) & 3 month testing window (30th Jun -> 26th Sep).
- ~38 weeks total: ~25 weeks training & ~13 weeks testing

## Payday

- Employees will get paid on the final friday of each month.

## Weekly Spending Loop

- Each week, each employee will have a specific weekly spend, which
  will VARY however this figure over many samples will calibrate to
  that employees personal weekly spend drawn from LAYER 1.
- Already built this -> call draw_weekly_spending_with_participation
  function. 
- HOWEVER, this function will ONLY give us the total weekly spend 
  on each category for a particular week, which isn't what I want 
  (limited data). Instead, I need to split this weekly spend into 
  several smaller purchases. For example, weekly spend on Food/Groceries
  could be £49 for specific account, HOWEVER, this isn't what you'd 
  see in real world banking data, some could frequently visit the 
  supermarket for grocery shopping so I need to split that weekly
  spend into smaller purchases.  

## Spending Frequency / Week (Category-Split)

Decided with two different transaction mechanisms:
- **CARD TRANSACTIONS** -> These are transactions which occur every week and sometimes more than once within a week e.g. transport, food etc.
- **DIRECT DEBIT TRANSACTIONS** -> These are transactions which occur on a monthly basis and are recurring payments e.g. utility bills. The categories subjected to these transactions are 'housing_fuel_power' & 'communication', with all items in these categories normally being paid monthly. 