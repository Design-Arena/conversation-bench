from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

end_session_function = FunctionSchema(
    name="end_session", description="End the current session.", properties={}, required=[]
)

search_products_function = FunctionSchema(
    name="search_products",
    description="Search for laptops matching specific criteria.",
    properties={
        "use_case": {
            "type": "string",
            "description": "Primary use case (e.g., 'coding', 'graphic_design', 'video_editing', 'general_student').",
        },
        "min_ram": {
            "type": "integer",
            "description": "Minimum RAM in GB.",
        },
        "min_storage": {
            "type": "integer",
            "description": "Minimum storage in GB.",
        },
        "max_price": {
            "type": "number",
            "description": "Maximum price in USD.",
        },
    },
    required=["use_case"],
)

compare_specs_function = FunctionSchema(
    name="compare_specs",
    description="Compare specifications of two or more products side by side.",
    properties={
        "product_ids": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of product IDs to compare (e.g., ['X1490', 'X1940']).",
        },
    },
    required=["product_ids"],
)

add_to_cart_function = FunctionSchema(
    name="add_to_cart",
    description="Add a product to the shopping cart with customer details.",
    properties={
        "product_id": {
            "type": "string",
            "description": "The product ID to add.",
        },
        "customer_name": {
            "type": "string",
            "description": "Customer's full name.",
        },
        "phone": {
            "type": "string",
            "description": "Customer's phone number.",
        },
        "condition": {
            "type": "string",
            "description": "Product condition: 'new' or 'open_box'.",
        },
        "warranty": {
            "type": "string",
            "description": "Warranty option: 'standard', 'extended', or 'accidental_damage'.",
        },
        "student_discount": {
            "type": "boolean",
            "description": "Whether to apply student discount.",
        },
    },
    required=["product_id", "customer_name", "phone", "condition"],
)

update_cart_function = FunctionSchema(
    name="update_cart",
    description="Update a field on an item in the shopping cart.",
    properties={
        "cart_id": {
            "type": "string",
            "description": "The cart ID to update.",
        },
        "field": {
            "type": "string",
            "description": "The field to update (e.g., 'product_id', 'condition', 'warranty', 'student_discount').",
        },
        "new_value": {
            "type": "string",
            "description": "The new value for the field.",
        },
    },
    required=["cart_id", "field", "new_value"],
)

check_student_discount_function = FunctionSchema(
    name="check_student_discount",
    description="Calculate the student discount price for a specific product.",
    properties={
        "product_id": {
            "type": "string",
            "description": "The product ID to check discount for.",
        },
    },
    required=["product_id"],
)

ToolsSchemaForTest = ToolsSchema(
    standard_tools=[
        end_session_function,
        search_products_function,
        compare_specs_function,
        add_to_cart_function,
        update_cart_function,
        check_student_discount_function,
    ]
)
