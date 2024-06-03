# Báo cáo kỹ thuật về cuộc thi xử lý dữ liệu tín dụng

## Introduction

From my prior experience with credit underwriting, I’ve come to appreciate that this is one of the most complex problems for applying machine learning. Data in this domain tends to be very heterogeneous, collected over different time frames, and coming from many different sources that may change and alter in the midst of the data collection process. Coming up with a proper target variable is also a very tricky process, requiring deep domain knowledge and refined business analysis skills. I want to commend Home Credit and Kaggle for providing such a great dataset, which was leak-free and very amenable to machine learning techniques.

Based on what is known about credit underwriting, and similar machine learning problems, it was clear that two things would be crucial for building a good model for this competition:
1. A good set of smart features.
2. A diverse set of base algorithms.

We were able to utilize four main sources of feature diversity, along with a few minor additional ones.
# Feature Engineer
Sau khi tham khảo nhiều lời giải từ các notebook giống như nhiều người em bắt đầu với một hàm feature_en đơn giản.Phương pháp này giúp tổng hợp và chuẩn hóa dữ liệu từ nhiều nguồn khác nhau, đồng thời xử lý các thông tin ngày tháng một cách hiệu quả để phục vụ cho các bước tiếp theo trong quy trình phân tích dữ liệu. Hàm này có nhiệm vụ là thực hiện feature engineering trên dữ liệu đầu vào. Hàm nhận một DataFrame cơ sở df_base và nhiều tập hợp các DataFrame bổ sung  (depth_0, depth_1, depth_2) kết hợp thêm xử lí ngày tháng sử dụng lớp Pipleline.
- **Quá trình**
- Thêm các đặc trưng mới vào DataFrame cơ sở:
    * `month_decision`: Trích xuất tháng từ cột "date_decision".
    * `weekday_decision`: Trích xuất ngày trong tuần từ cột "date_decision".

- Lặp qua tổng hợp của ba danh sách DataFrame bổ sung (`depth_0 + depth_1 + depth_2`). Trong mỗi vòng lặp, DataFrame hiện tại được kết hợp (join) với `df_base` theo cột khóa `case_id` với phương pháp left join. Một hậu tố `_i` được thêm vào tên cột của DataFrame hiện tại để phân biệt với các cột đã có trong `df_base`.

- Thực hiện xử lý ngày tháng sử dụng lớp `Pipeline` bằng cách áp dụng phương thức `handle_dates`.

- Trả về DataFrame đã được thực hiện feature engineering.

## Tối ưu bộ nhớ
Sau khi tham khảo lời giải từ notebook của XIAOLEI LIAN cũng như nhiều notebook khác, em biết được rằng khi làm việc với dữ liệu lớn vốn tiêu thụ lượng lớn RAM thì việc giảm thiểu bộ nhớ sử dụng của dữ liệu là vô cùng quan trọng. Để làm được điều đó thì đối với các dữ liệu số nguyên, số thực thì em sẽ tìm các giá trị lớn nhất và nhỏ nhất trong cột đó và tìm cách giảm kiểu dữ liệu xuống để dữ liệu không bị tràn số, chẳng hạn như từ int64 xuống còn int32. Còn đối với kiểu dữ liệu category thì bỏ qua vì vốn dĩ nó đã tối ưu bộ nhớ rồi. Tương tự với kiểu dữ liệu object. Trước khi tối ưu bộ nhớ, tập train tiêu thụ 4322.75 MB, con số này sau khi xử lí giảm bộ nhớ thì còn 1528.81 MB. Đối với tập test, con số này lần lượt là 0.04 MB và 0.02 MB. 
