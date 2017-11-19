(ns ml.core
  (:require [ml.helpers :as helpers]))

(defn ready-function
  [w-first-layer w-second-layer]
  (fn [x]
    (let [hidden-units (map #(helpers/basis-function :neural-network {:x x :w % :activation-fn :tanh})
                            w-first-layer)]
      (helpers/linear-model :regression {:w (first w-second-layer) :basis-fns hidden-units :activation-fn :identity}))))

(defmulti neural-network
  (fn [_ {method :method}]
    method))

(defmethod neural-network :feed-forward
  [observations
   {:keys [max-iterations learning-rate] :as options}]
  (let [vector-valued-observation? (vector? (:x (first observations)))
        max-iterations (or max-iterations 100)
        M 4
        learning-rate (or learning-rate 0.4)]
    (reduce (fn [layer-weights {:keys [x t] :as observation}]
              (let [x (conj x 1)]
                (loop [i 1
                       w-first-layer (if (empty? layer-weights)
                                       (into [] (repeat M
                                                        (vec (take (count x)
                                                                   (iterate inc 1)))))
                                       (first layer-weights))
                       w-second-layer (if (empty? layer-weights)
                                        (into [] (repeat (count t)
                                                         (vec (take M
                                                                    (iterate inc 1)))))
                                        (second layer-weights))
                       gradient 100]
                  (if (or (> i max-iterations)
                          (< gradient (helpers/abs 0.008)))
                    (let [ready-function (ready-function w-first-layer w-second-layer)
                          y-diff (mapv #(- (first (:t %)) (ready-function (:x %))) observations)]
                      (println "gradient " (pr-str gradient))
                      (println "---- READY ----> " (pr-str y-diff))
                      (println "--------------------------")
                      [w-first-layer w-second-layer])
                    (let [hidden-units (map #(helpers/basis-function :neural-network {:x x :w % :activation-fn :tanh})
                                            w-first-layer)
                          y (map #(helpers/linear-model :regression {:w % :basis-fns hidden-units :activation-fn :identity})
                                 w-second-layer)
                          output-errors (map #(- %1 %2) y t)
                          hidden-unit-errors (map-indexed (fn [index hidden-unit]
                                                            (helpers/error-backpropagate :tanh {:hidden-unit hidden-unit
                                                                                                :output-errors output-errors
                                                                                                :w (helpers/column w-second-layer index)}))
                                                          hidden-units)
                          first-layer-error-derivatives (mapv (fn [error]
                                                                (mapv #(* error %)
                                                                      x))
                                                              hidden-unit-errors)
                          second-layer-error-derivatives (mapv (fn [error]
                                                                 (mapv #(* error %)
                                                                       hidden-units))
                                                               output-errors)
                          gradient (apply + (mapcat #(concat %1 %2) first-layer-error-derivatives second-layer-error-derivatives))]
                      (recur (inc i)
                             (helpers/M-minus w-first-layer (helpers/M-element-multiply learning-rate first-layer-error-derivatives))
                             (helpers/M-minus w-second-layer (helpers/M-element-multiply learning-rate second-layer-error-derivatives))
                             gradient))))))
         [] observations)))