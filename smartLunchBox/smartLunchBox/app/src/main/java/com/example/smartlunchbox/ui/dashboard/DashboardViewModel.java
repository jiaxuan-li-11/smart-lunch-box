package com.example.smartlunchbox.ui.dashboard;

import static android.graphics.Typeface.BOLD;
import static android.graphics.Typeface.DEFAULT_BOLD;
import static android.graphics.Typeface.ITALIC;
import static android.graphics.Typeface.defaultFromStyle;

import android.graphics.Color;
import android.text.SpannableString;
import android.text.style.ForegroundColorSpan;
import android.text.style.RelativeSizeSpan;
import android.text.style.StyleSpan;
import android.util.Log;

import androidx.lifecycle.ViewModel;

import com.github.mikephil.charting.charts.BarChart;
import com.github.mikephil.charting.charts.PieChart;
import com.github.mikephil.charting.components.Description;
import com.github.mikephil.charting.components.Legend;
import com.github.mikephil.charting.components.XAxis;
import com.github.mikephil.charting.data.BarData;
import com.github.mikephil.charting.data.BarDataSet;
import com.github.mikephil.charting.data.BarEntry;
import com.github.mikephil.charting.data.PieData;
import com.github.mikephil.charting.data.PieDataSet;
import com.github.mikephil.charting.data.PieEntry;
import com.github.mikephil.charting.formatter.ValueFormatter;
import com.github.mikephil.charting.utils.ColorTemplate;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Date;
import java.util.TimeZone;

public class DashboardViewModel extends ViewModel {


    public void pieChart(PieChart pieChart, float usedCal, float dayLeft){

        pieChart.setHoleRadius(60f);
        pieChart.getDescription().setEnabled(false);
        pieChart.setDrawCenterText(true);
        pieChart.setCenterText(centerSpannableText());
        pieChart.setCenterTextSize(22);
        pieChart.setCenterTextTypeface(defaultFromStyle(BOLD));

        ArrayList<PieEntry> calorieList = new ArrayList<>();
        calorieList.add(new PieEntry(usedCal, "Calories Eaten"));
        calorieList.add(new PieEntry(dayLeft, "Today Cal Left"));

        ArrayList<String> nutrition = new ArrayList<>();

        nutrition.add("Calories Eaten");
        nutrition.add("Today Cal Left");

        PieDataSet dataSet = new PieDataSet(calorieList, "");

        ArrayList<Integer> colors = new ArrayList<>();
        colors.add(Color.parseColor("#ff9900"));
        colors.add(Color.parseColor("#e4e4e4"));
//        colors.add(Color.parseColor("#ff9900"));

        dataSet.setColors(colors);

        Legend legend = pieChart.getLegend();
        legend.setForm(Legend.LegendForm.CIRCLE);
        legend.setVerticalAlignment(Legend.LegendVerticalAlignment.BOTTOM);
        legend.setHorizontalAlignment(Legend.LegendHorizontalAlignment.CENTER);
        legend.setOrientation(Legend.LegendOrientation.HORIZONTAL);
        legend.setTextSize(10f);
        legend.setFormSize(10f);
        legend.setWordWrapEnabled(true);

        PieData pieData = new PieData(dataSet);

        pieData.setValueTextSize(14);
        pieData.setValueTypeface(DEFAULT_BOLD);
        pieData.setValueTextColor(Color.parseColor("#000000"));

        pieChart.setData(pieData);
        pieChart.setDrawEntryLabels(false);
        pieChart.invalidate();
        pieChart.notifyDataSetChanged();
    }

    public void barChart(BarChart barChart, ArrayList<Float> barList){

        Log.d("date", barList.toString());

        ArrayList<BarEntry> barEntries = new ArrayList<>();
        barEntries.add(new BarEntry(0, barList.get(0)));
        barEntries.add(new BarEntry(1, barList.get(1)));
        barEntries.add(new BarEntry(2, barList.get(2)));
        barEntries.add(new BarEntry(3, barList.get(3)));
        barEntries.add(new BarEntry(4, barList.get(4)));
        barEntries.add(new BarEntry(5, barList.get(5)));
        barEntries.add(new BarEntry(6, barList.get(6)));


        Legend legend = barChart.getLegend();
        legend.setEnabled(false);

        BarDataSet barDataSet = new BarDataSet(barEntries, "");
        barDataSet.setColors(ColorTemplate.MATERIAL_COLORS);
        barDataSet.setValueTextColor(Color.BLACK);
        barDataSet.setValueTextSize(10f);
        barDataSet.setFormLineWidth(1f);
        barDataSet.setFormSize(10f);

        BarData barData = new BarData(barDataSet);

        XAxis xAxis = barChart.getXAxis();
        xAxis.setAxisMinimum(-1);
        xAxis.setAxisMaximum(7);
        xAxis.setDrawGridLines(false);

        ArrayList<String> dateList = new ArrayList<String>();
        for (int i = 1; i < 8; i++){
            dateList.add(getNDate(7-i));
        }
        int size = dateList.size();
        if (size >= 0 && size <= 7){
            xAxis.setLabelCount(size, false);
        }else{
            xAxis.setLabelCount(7, true);
        }
        xAxis.setGranularity(1);
        xAxis.setGranularityEnabled(true);
        xAxis.setPosition(XAxis.XAxisPosition.BOTTOM);

        Log.d("---", dateList.toString());
        xAxis.setValueFormatter(new ValueFormatter() {
            @Override
            public String getFormattedValue(float value) {
                int values = (int) value;
                if(values >= 0 && values < size){
                    return dateList.get(values);
                }else{
                    return "";
                }
            }
        });


        barChart.getAxisLeft().setEnabled(false);
        barChart.getAxisRight().setEnabled(false);

        Description description = new Description();
        description.setEnabled(false);
        barChart.setDescription(description);

        barChart.setData(barData);
        barChart.getXAxis().setLabelRotationAngle(-60);
        barDataSet.setColors(ColorTemplate.COLORFUL_COLORS);

        barChart.invalidate();
        barChart.notifyDataSetChanged();


    }

    private String getNDate(int N){
        Calendar calendar = Calendar.getInstance(TimeZone.getTimeZone("EST"));
        calendar.add(Calendar.DAY_OF_YEAR, -N);
        Date newDate = calendar.getTime();
        SimpleDateFormat s = new SimpleDateFormat("MM-dd");
        return s.format(newDate);
    }


    private SpannableString centerSpannableText() {
        SpannableString str = new SpannableString("Calories\nFor Today");
        str.setSpan(new ForegroundColorSpan(Color.parseColor("#151515")), 0, 7, str.SPAN_EXCLUSIVE_EXCLUSIVE);
        str.setSpan(new StyleSpan(BOLD), 0, 7, str.SPAN_EXCLUSIVE_EXCLUSIVE);
        str.setSpan(new ForegroundColorSpan(Color.parseColor("#767676")), 8, 18, str.SPAN_EXCLUSIVE_EXCLUSIVE);
        str.setSpan(new RelativeSizeSpan(0.5f), 8, 18, str.SPAN_EXCLUSIVE_EXCLUSIVE);
        str.setSpan(new StyleSpan(ITALIC), 8, 18, str.SPAN_EXCLUSIVE_EXCLUSIVE);

        return str;
    }


}