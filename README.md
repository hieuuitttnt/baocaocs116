# Báo cáo kỹ thuật về cuộc thi xử lý dữ liệu tín dụng

## Giới Thiệu
Từ kinh nghiệm trước đây của em trong việc thẩm định tín dụng, em nhận thấy rằng đây là một trong những vấn đề phức tạp nhất để áp dụng học máy. Dữ liệu trong lĩnh vực này thường rất đa dạng, được thu thập qua các khoảng thời gian khác nhau và đến từ nhiều nguồn khác nhau, có thể thay đổi và thay thế trong quá trình thu thập dữ liệu. Việc đưa ra một biến mục tiêu thích hợp cũng là một quá trình rất khó khăn, đòi hỏi kiến thức sâu rộng về lĩnh vực này và kỹ năng phân tích kinh doanh tinh tế. Em muốn khen ngợi Home Credit và Kaggle vì đã cung cấp một bộ dữ liệu tuyệt vời như vậy, không có rò rỉ và rất dễ dàng áp dụng các kỹ thuật học máy.

Dựa trên những gì đã biết về thẩm định tín dụng và các vấn đề học máy tương tự, rõ ràng hai điều sẽ rất quan trọng để xây dựng một mô hình tốt cho cuộc thi này:

Một tập hợp các đặc trưng thông minh.
Một tập hợp đa dạng các thuật toán cơ bản.
Em đã có thể tận dụng bốn nguồn chính của sự đa dạng đặc trưng, cùng với một số nguồn nhỏ bổ sung khác.

## Trình tổng hợp cho trích xuất đặc trưng  (Aggregator for Feature Extraction)
__Tổng quan__

Nhóm chúng em đã tham khảo một số notebook để cho ra class Aggregator định nghĩa bao gồm nhiều phương thức để trích xuất và tính toán các giá trị thống kê từ một DataFrame (df). Mỗi phương thức tập trung vào một loại dữ liệu cụ thể dựa trên hậu tố của tên cột và tạo ra các biểu thức tính toán khác nhau như giá trị lớn nhất (max), giá trị cuối cùng (last), giá trị trung bình (mean)

__Chi tiết__ 

**1. Phương thức num_expr(df)**

Mục đích: Trích xuất các cột 'số học' từ DataFrame.
Lựa chọn cột: Chọn các cột có tên kết thúc bằng "P" hoặc "A".

Biểu thức tạo ra:

    + expr_max: Tính giá trị lớn nhất của mỗi cột và đặt alias là max_{col}.

    + expr_last: Tính giá trị cuối cùng của mỗi cột và đặt alias là last_{col}.
    
    + expr_mean: Tính giá trị trung bình của mỗi cột và đặt alias là mean_{col}.

Cuối cùng, trả danh sách các biểu thức tính toán 'expr_max + expr_last + expr_mean' các giá trị thống kê cho các cột "chuỗi" được chọn từ DataFrame. 
 

**2. Phương thức num_expr(df)**

Mục đích: Trích xuất các cột 'ngày' từ DataFrame.
Lựa chọn cột: Chọn các cột có tên kết thúc bằng "D".

Biểu thức tạo ra:

    + expr_max: Tính giá trị lớn nhất của mỗi cột và đặt alias là max_{col}.

    + expr_last: Tính giá trị cuối cùng của mỗi cột và đặt alias là last_{col}.
    
    + expr_mean: Tính giá trị trung bình của mỗi cột và đặt alias là mean_{col}.

Cuối cùng, trả danh sách các biểu thức tính toán 'expr_max + expr_last + expr_meam' các giá trị thống kê cho các cột "ngày" được chọn từ DataFrame. 

**3. Phương thức str_expr(df)**

Mục đích: Trích xuất các cột 'chuỗi' từ DataFrame.
Lựa chọn cột: Chọn các cột có tên kết thúc bằng "M".

Biểu thức tạo ra:

    + expr_max: Tính giá trị lớn nhất của mỗi cột và đặt alias là max_{col}.

    + expr_last: Tính giá trị cuối cùng của mỗi cột và đặt alias là last_{col}.

Cuối cùng, trả danh sách các biểu thức tính toán 'expr_max + expr_last' các giá trị thống kê cho các cột "chuỗi" được chọn từ DataFrame. 

**4. Phương thức str_expr(df)**

Mục đích: Trích xuất các cột 'khác' từ DataFrame.
Lựa chọn cột: Chọn các cột có tên kết thúc bằng "T" or "L".

Biểu thức tạo ra:

    + expr_max: Tính giá trị lớn nhất của mỗi cột và đặt alias là max_{col}.

    + expr_last: Tính giá trị cuối cùng của mỗi cột và đặt alias là last_{col}.

Cuối cùng, trả danh sách các biểu thức tính toán 'expr_max + expr_last' các giá trị thống kê cho các cột "khác" được chọn từ DataFrame. 

**5. Phương thức count_expr(df)**


Mục đích: Trích xuất các cột 'đếm' từ DataFrame.
Lựa chọn cột: Chọn các cột có "num_group" trong tên.

Biểu thức tạo ra:

    + expr_max: Tính giá trị lớn nhất của mỗi cột và đặt alias là max_{col}.

    + expr_last: Tính giá trị cuối cùng của mỗi cột và đặt alias là last_{col}.

 Cuối cùng, trả danh sách các biểu thức tính toán 'expr_max + expr_last' các giá trị thống kê cho các cột "đếm" được chọn từ DataFrame. 

**6. Phương thức get(df)**

Mục đích: Phương thức này tổng hợp tất cả các biểu thức từ các phương thức trước đó để có được danh sách đầy đủ các biểu thức trích xuất đặc điểm.

Nó gọi tất cả các phương pháp trích xuất đặc trưng riêng lẻ và nối các danh sách kết quả.

Trả về danh sách biểu thức tổng hợp cho tất cả các loại đối tượng.

__Nhóm cũng đã thử nghiệm và nhận thấy rằng khi chỉ sử dụng giá trị lớn nhất (maximum value) để trích xuất đặc trưng, kết quả điểm số bị giảm đi đáng kể so với khi sử dụng kết hợp nhiều biểu thức khác nhau như giá trị cuối cùng và giá trị trung bình.__



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
## Base model
In the context of your provided file, which appears to involve the use of LightGBM (LGB) and CatBoost models, here's how these concepts apply:

Base Models in the File:

LightGBM: A gradient boosting framework that uses tree-based learning algorithms. It is known for its efficiency and speed, particularly with large datasets.
CatBoost: Another gradient boosting algorithm that is particularly well-suited for categorical data and can handle complex datasets effectively.
## Ensemble
Trọng số của Mô hình và Voting:
Lớp VotingModel định nghĩa cách kết hợp các dự đoán từ các mô hình riêng lẻ.

Trọng số: em cho tầm quan trọng bằng nhau cho các dự đoán của mỗi mô hình.Vì không có mô hình nào nổi bật hơn hẳn trong việc predict.Em cũng thử nghiệm đánh lại trọng số cho từng mô hình nhưng kết quả không khả quan.
Ví dụ: Nếu chúng ta có 5 mô hình LightGBM và 5 mô hình CatBoost, mỗi dự đoán của mô hình được gán trọng số là 1/10.
Kết hợp Dự đoán:

Phương pháp predict_proba xử lý việc kết hợp các xác suất:
Các Tính năng Phân loại: Nó xử lý các tính năng phân loại khác nhau giữa các mô hình LightGBM và CatBoost. Điều này quan trọng vì các loại mô hình khác nhau có thể xử lý các tính năng phân loại theo cách khác nhau.
Trung bình Xác suất: Xác suất cuối cùng được tính toán là trung bình của các xác suất từ tất cả các mô hình.
## Các thử nghiệm thất bại

