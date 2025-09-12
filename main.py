import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    print("hi")
    return


if __name__ == "__main__":
    app.run()
