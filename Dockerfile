FROM python 3.9-slim
WORKDIR /Plastics_Image_Classification
COPY ./Plastics_Image_Classifications
RUN pip install --no-cach-dir -r requirements.txt
EXPOSE 7860
CMD ["python", "Plastics_Image_Classification"]
