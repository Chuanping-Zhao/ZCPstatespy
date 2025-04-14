def style_df(df,backcolor="#2A8A9A",fontcolor="white"):
    return df.style.set_properties(**{
        "text-align": "center",
        "background-color": backcolor,#D63A79
        "color": fontcolor,
        "border": "1.5px solid black"
    }).set_table_styles([
        {"selector": "th", "props": [("text-align", "center")]}
    ])