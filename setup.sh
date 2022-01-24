mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
[theme]\n\
primaryColor="#f9b36b"
backgroundColor="#f3e9ce"
secondaryBackgroundColor="#f9b36b"
textColor="#000000"
" 
> ~/.streamlit/config.toml
