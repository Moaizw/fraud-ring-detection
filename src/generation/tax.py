"""
This file will convert gross annual income to take home annual income 
using UK 2025/26 Income Tax and National Insurance rules 
(England/Wales/NI only; Scotland has separate income tax bands, 
deliberately out of scope, documented as a simplification).

Includes the personal allowance taper (income above £100,000 loses £1 of
allowance per £2 earned, down to £0 by £125,140), relevant here since
Managers/Senior Professional rows in the simulation regularly produce
six-figure gross incomes.
"""

PERSONAL_ALLOWANCE = 12570
TAPER_THRESHOLD = 100000
TAPER_LIMIT = 125140  

BASIC_RATE_LIMIT = 50270
HIGHER_RATE_LIMIT = 125140

INCOME_TAX_BASIC_RATE = 0.20
INCOME_TAX_HIGHER_RATE = 0.40
INCOME_TAX_ADDITIONAL_RATE = 0.45

NI_MAIN_RATE = 0.08
NI_UPPER_RATE = 0.02


def calculate_personal_allowance(gross_annual: float) -> float:
    """
    Calculates actual personal allowance -> IMPORTANT FOR those earning > 100,000.
    """

    if gross_annual >= TAPER_LIMIT:
        return 0
    elif gross_annual <= TAPER_THRESHOLD:
        return PERSONAL_ALLOWANCE
    else:
        return PERSONAL_ALLOWANCE - (gross_annual - TAPER_THRESHOLD) / 2


def calculate_income_tax(gross_annual: float) -> float:
    """
    Calculate UK Income Tax (England/Wales/NI, 2025/26), banded, using the actual
    personal allowance as the tax-free threshold.
    """

    allowance = calculate_personal_allowance(gross_annual)

    if gross_annual <= BASIC_RATE_LIMIT:
        if gross_annual <= allowance: #those that earn LESS than personal income allowance
            return 0.0
        else:
            return INCOME_TAX_BASIC_RATE * (gross_annual - allowance) #earn between £12571 - £50270
    elif BASIC_RATE_LIMIT < gross_annual <= HIGHER_RATE_LIMIT:
        tax_20 = INCOME_TAX_BASIC_RATE * (BASIC_RATE_LIMIT - allowance)
        tax_40 = INCOME_TAX_HIGHER_RATE * (gross_annual - BASIC_RATE_LIMIT)
        return tax_20 + tax_40 
    else:
        tax_20 = INCOME_TAX_BASIC_RATE * (BASIC_RATE_LIMIT - allowance) #allowance completely gone for these earners
        tax_40 = INCOME_TAX_HIGHER_RATE * (HIGHER_RATE_LIMIT - BASIC_RATE_LIMIT)
        tax_45 = INCOME_TAX_ADDITIONAL_RATE * (gross_annual - HIGHER_RATE_LIMIT)
        return tax_20 + tax_40 + tax_45


def calculate_ni(gross_annual: float) -> float:
    """
    UK Class 1 National Insurance (employee), 2025/26, banded. Uses the
    FIXED PERSONAL_ALLOWANCE as its threshold.
    NI NOT AFFECTED by 'personal allowance taper' seen in income tax.
    """

    if gross_annual <= PERSONAL_ALLOWANCE:
        return 0.0
    elif gross_annual <= BASIC_RATE_LIMIT:
        return NI_MAIN_RATE * (gross_annual - PERSONAL_ALLOWANCE)
    else:
        band_8 = NI_MAIN_RATE * (BASIC_RATE_LIMIT - PERSONAL_ALLOWANCE)
        band_2 = NI_UPPER_RATE * (gross_annual - BASIC_RATE_LIMIT)
        return band_8 + band_2


def gross_to_net(gross_annual: float) -> float:
    """
    Convert gross annual income to net annual income.
    """
    
    income_tax = calculate_income_tax(gross_annual)
    ni = calculate_ni(gross_annual)
    return gross_annual - income_tax - ni


if __name__ == "__main__":
    #Expected: £11,000 -> £11,000, £39,000 → £31,599.60, £99,000 → £67,977.40, £110,000 → £73,357.40, £140,000 → £88,500.40
    for income in [11000, 39000, 99000, 110000, 140000]: 
        print(f"Gross: £{income:,} -> Net: £{gross_to_net(income):,.2f}")

    
    

    