"""
All categories currently used in the app
"""

ALL_CATEGORIES_LIST = [
  'appointment_letter',
  'bank_statement',
  'birth_certificate',
  'electricity_bill',
  'form_3',
  'form_4',
  'form_a',
  'introductory_letter',
  'mortgage_statement',
  'payslip',
  'property_rate',
  'rejected',
  'ssnit_pension_statement',
  'tenancy_agreement',
  'water_bill'
]

CATEGORIES_AND_TYPES_DICT = {
  'POA': [
    'bank_statement', 'electricity_bill',
    'form_3', 'form_4', 'form_a',
    'mortgage_statement', 'property_rate',
    'tenancy_agreement', 'water_bill',
  ],
  'POE': [
    'appointment_letter', 'form_a', 'introductory_letter',
    'payslip', 'ssnit_pension_statement',
  ]
}
